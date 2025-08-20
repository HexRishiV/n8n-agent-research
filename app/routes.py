import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

router = APIRouter()

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = None

class ArticleMetadata(BaseModel):
    tags: List[str]
    sentiment: str

class Article(BaseModel):
    id: str
    summary: str
    metadata: ArticleMetadata
    date_fetched: Optional[str] = None

class ArticlesRequest(BaseModel):
    articles: List[Article]

class SearchRequest(BaseModel):
    query: str
    max_results: int = 6
    
def clean_llm_json_response(raw_text: str) -> List[Dict]:
    """
    Clean LLM response that might be wrapped in markdown or have extra text
    """
    try:
        code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
        match = re.search(code_block_pattern, raw_text, re.DOTALL)
        
        if match:
            json_text = match.group(1).strip()
        else:
            json_text = raw_text.strip()
        
        json_pattern = r'\[.*?\]'
        json_match = re.search(json_pattern, json_text, re.DOTALL)
        
        if json_match:
            json_text = json_match.group(0)
        
        parsed_data = json.loads(json_text)
        
        if isinstance(parsed_data, dict):
            parsed_data = [parsed_data]
        elif not isinstance(parsed_data, list):
            raise ValueError("Not a valid array")
            
        return parsed_data
        
    except Exception as e:
        print(f"Error cleaning JSON: {e}")
        print(f"Raw text: {raw_text[:500]}...")
        return []

@router.post("/api/articles/organize-flexible")
async def organize_articles_flexible(raw_data: Union[str, dict, list]):
    """
    Flexible endpoint that handles markdown-wrapped JSON from LLMs
    """
    try:
        base_dir = "saved_news"
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        articles_data = []
        
        if isinstance(raw_data, str):
            articles_data = clean_llm_json_response(raw_data)
        elif isinstance(raw_data, list):
            articles_data = raw_data
        elif isinstance(raw_data, dict):
            if "output" in raw_data:
                output_text = raw_data["output"]
                if isinstance(output_text, str):
                    articles_data = clean_llm_json_response(output_text)
                else:
                    articles_data = output_text if isinstance(output_text, list) else [output_text]
            elif "articles" in raw_data:
                articles_data = raw_data["articles"]
            else:
                articles_data = [raw_data]
        
        if not articles_data:
            raise HTTPException(status_code=400, detail="No valid articles found in input")
        
        organized_count = 0
        current_time = datetime.now().isoformat()
        
        for article_data in articles_data:
            try:
                article_id = article_data.get("id", f"auto_{abs(hash(str(article_data)))}")
                
                summary = article_data.get("summary", "")
                if not summary and "title" in article_data:
                    summary = article_data["title"]
                if not summary:
                    summary = "No summary available"
                
                metadata = article_data.get("metadata", {})
                
                tags = metadata.get("tags", [])
                if not tags:
                    content_lower = summary.lower()
                    potential_tags = []
                    
                    if any(word in content_lower for word in ["tech", "ai", "computer", "digital"]):
                        potential_tags.append("technology")
                    if any(word in content_lower for word in ["health", "medical", "hospital", "doctor"]):
                        potential_tags.append("healthcare")
                    if any(word in content_lower for word in ["politics", "government", "election", "policy"]):
                        potential_tags.append("politics")
                    if any(word in content_lower for word in ["sports", "game", "team", "player"]):
                        potential_tags.append("sports")
                    if any(word in content_lower for word in ["business", "economy", "financial", "market"]):
                        potential_tags.append("business")
                    
                    tags = potential_tags if potential_tags else ["news"]
                
                sentiment = metadata.get("sentiment", "")
                if not sentiment:
                    content_lower = summary.lower()
                    positive_words = ["rise", "up", "growth", "success", "win", "positive", "good", "great"]
                    negative_words = ["fall", "down", "loss", "fail", "negative", "bad", "crisis", "death"]
                    
                    pos_count = sum(1 for word in positive_words if word in content_lower)
                    neg_count = sum(1 for word in negative_words if word in content_lower)
                    
                    sentiment = "positive" if pos_count > neg_count else "negative"
                
                date_fetched = article_data.get("date_fetched", current_time)
                
                for tag in tags:
                    tag_dir = os.path.join(base_dir, tag.lower().replace(" ", "_"))
                    if not os.path.exists(tag_dir):
                        os.makedirs(tag_dir)
                    
                    filename = f"{article_id}.json"
                    filepath = os.path.join(tag_dir, filename)
                    
                    clean_data = {
                        "id": article_id,
                        "summary": summary,
                        "metadata": {
                            "tags": tags,
                            "sentiment": sentiment
                        },
                        "date_fetched": date_fetched
                    }
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(clean_data, f, indent=2, ensure_ascii=False)
                    
                    organized_count += 1
                    
            except Exception as e:
                print(f"Error processing article: {article_data}, Error: {e}")
                continue
        
        return {
            "status": "success",
            "message": f"Organized {len(articles_data)} articles into {organized_count} files",
            "base_directory": base_dir,
            "articles_processed": len(articles_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error organizing articles: {str(e)}")


@router.post("/api/articles/organize")
async def organize_articles(request: ArticlesRequest):
    """
    Organize articles by tags into directory structure
    """
    try:
        base_dir = "saved_news"
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        organized_count = 0
        current_time = datetime.now().isoformat()
        
        for article in request.articles:
            for tag in article.metadata.tags:
                tag_dir = os.path.join(base_dir, tag.lower().replace(" ", "_"))
                if not os.path.exists(tag_dir):
                    os.makedirs(tag_dir)
                
                filename = f"{article.id}.json"
                filepath = os.path.join(tag_dir, filename)
                
                article_data = {
                    "id": article.id,
                    "summary": article.summary,
                    "metadata": {
                        "tags": article.metadata.tags,
                        "sentiment": article.metadata.sentiment
                    },
                    "date_fetched": article.date_fetched or current_time
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(article_data, f, indent=2, ensure_ascii=False)
                
                organized_count += 1
        
        return {
            "status": "success",
            "message": f"Organized {len(request.articles)} articles into {organized_count} files",
            "base_directory": base_dir
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error organizing articles: {str(e)}")

def load_articles_from_directory(directory_path: str) -> List[Dict]:
    """Load all articles from a directory"""
    articles = []
    if not os.path.exists(directory_path):
        return articles
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                    articles.append(article_data)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
    
    return articles

def create_index_from_articles(articles: List[Dict]) -> VectorStoreIndex:
    """Create a LlamaIndex from articles"""
    documents = []
    
    for article in articles:
        content = f"{article['summary']} Tags: {', '.join(article['metadata']['tags'])} Sentiment: {article['metadata']['sentiment']}"
        
        doc = Document(
            text=content,
            metadata={
                "id": article["id"],
                "summary": article["summary"],
                "tags": article["metadata"]["tags"],
                "sentiment": article["metadata"]["sentiment"],
                "date_fetched": article.get("date_fetched", "")
            }
        )
        documents.append(doc)
    
    if not documents:
        return None
    
    index = VectorStoreIndex.from_documents(documents)
    return index

@router.post("/api/articles/search")
async def search_articles(search_request: SearchRequest):
    """
    Search for relevant articles based on query
    """
    try:
        base_dir = "saved_news"
        query = search_request.query.lower()
        max_results = min(search_request.max_results, 10)
        
        potential_tag = query.replace(" ", "_")
        tag_dir = os.path.join(base_dir, potential_tag)
        
        articles = []
        search_scope = "unknown"
        
        if os.path.exists(tag_dir):
            articles = load_articles_from_directory(tag_dir)
            search_scope = f"tag folder: {potential_tag}"
        else:
            if os.path.exists(base_dir):
                for tag_folder in os.listdir(base_dir):
                    tag_path = os.path.join(base_dir, tag_folder)
                    if os.path.isdir(tag_path):
                        folder_articles = load_articles_from_directory(tag_path)
                        articles.extend(folder_articles)
            search_scope = "all articles"
        
        if not articles:
            return {
                "query": search_request.query,
                "results": [],
                "total_found": 0,
                "search_scope": search_scope,
                "message": "No articles found"
            }
        
        index = create_index_from_articles(articles)
        if not index:
            return {
                "query": search_request.query,
                "results": [],
                "total_found": 0,
                "search_scope": search_scope,
                "message": "Could not create search index"
            }
        
        retriever = index.as_retriever(similarity_top_k=max_results)
        nodes = retriever.retrieve(search_request.query)
        
        relevant_articles = []
        for node in nodes:
            article_data = {
                "id": node.metadata["id"],
                "summary": node.metadata["summary"],
                "metadata": {
                    "tags": node.metadata["tags"],
                    "sentiment": node.metadata["sentiment"]
                },
                "date_fetched": node.metadata["date_fetched"],
                "relevance_score": node.score
            }
            relevant_articles.append(article_data)
        
        return {
            "query": search_request.query,
            "results": relevant_articles,
            "total_found": len(relevant_articles),
            "search_scope": search_scope,
            "total_articles_searched": len(articles)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@router.get("/api/articles/search-simple")
async def search_articles_simple(
    q: str = Query(..., description="Search query"),
    max_results: int = Query(6, ge=1, le=10, description="Maximum number of results")
):
    """
    Simple GET endpoint for searching articles
    """
    search_request = SearchRequest(query=q, max_results=max_results)
    return await search_articles(search_request)

@router.get("/api/articles/tags")
async def get_available_tags():
    """
    Get list of all available tags (directories)
    """
    base_dir = "saved_news"
    
    if not os.path.exists(base_dir):
        return {"tags": []}
    
    tags = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    return {"tags": tags}

@router.get("/api/articles/by-tag/{tag}")
async def get_articles_by_tag(tag: str):
    """
    Get all articles for a specific tag
    """
    tag_dir = os.path.join("saved_news", tag.lower().replace(" ", "_"))
    
    if not os.path.exists(tag_dir):
        raise HTTPException(status_code=404, detail=f"Tag '{tag}' not found")
    
    articles = load_articles_from_directory(tag_dir)
    
    return {"tag": tag, "articles": articles, "count": len(articles)}

@router.get("/api/articles/stats")
async def get_article_stats():
    """
    Get statistics about stored articles
    """
    base_dir = "saved_news"
    
    if not os.path.exists(base_dir):
        return {"total_articles": 0, "total_tags": 0, "tags": []}
    
    total_articles = 0
    tag_stats = []
    
    for tag_folder in os.listdir(base_dir):
        tag_path = os.path.join(base_dir, tag_folder)
        if os.path.isdir(tag_path):
            articles = load_articles_from_directory(tag_path)
            article_count = len(articles)
            total_articles += article_count
            
            tag_stats.append({
                "tag": tag_folder,
                "article_count": article_count
            })
    
    return {
        "total_articles": total_articles,
        "total_tags": len(tag_stats),
        "tag_stats": sorted(tag_stats, key=lambda x: x["article_count"], reverse=True)
    }