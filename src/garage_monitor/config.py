from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GM_")

    blink_username: str
    blink_password: str
    blink_camera_name: str = "Garage"
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    confidence_threshold: float = 0.7
    telegram_bot_token: str
    telegram_chat_id: str
    gcp_project_id: str
    firestore_collection: str = "garage_monitor"
