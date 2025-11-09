from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import chat, documents, sentiment, metrics, data

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix=f"{settings.API_V1_PREFIX}/chat", tags=["chat"])
app.include_router(documents.router, prefix=f"{settings.API_V1_PREFIX}/documents", tags=["documents"])
app.include_router(sentiment.router, prefix=f"{settings.API_V1_PREFIX}/sentiment", tags=["sentiment"])
app.include_router(metrics.router, prefix=f"{settings.API_V1_PREFIX}/metrics", tags=["metrics"])
app.include_router(data.router, prefix=f"{settings.API_V1_PREFIX}/data", tags=["data"])


@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
