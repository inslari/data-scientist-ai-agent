from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.database import Base, engine
from api.models.mcp import MCPConfigModel
from api.models.csv import CSVFileModel
from api.models.data_group import DataGroupModel
from api.models.project import ProjectModel

# Veritabanı tablolarını oluştur
Base.metadata.create_all(bind=engine)

from api.routes.mcp import router as mcp_router
from api.routes.csv import router as csv_router
from api.routes.data_group import router as data_group_router
from api.routes.project import router as project_router

app = FastAPI(
    title="Data Scientist AI Agent API",
    description="API for the Data Scientist AI Agent",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(data_group_router)
app.include_router(project_router)
app.include_router(mcp_router)
app.include_router(csv_router)

@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "ok", "message": "API is running successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
