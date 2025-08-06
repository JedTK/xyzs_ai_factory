from xyzs_py.database import get_read_session

from xyzs_ai_factory.entity.OpenAIConfigEntity import OpenAIConfigEntity
from xyzs_py.XLogs import XLogs
from xyzs_py.cache.XCacheFactory import XCacheFactory

# 记录日志
logger = XLogs(__name__)


class OpenAIConfigService:
    """
    OpenAIConfig服务类，用于数据库查询
    """

    @classmethod
    def get_config(cls, config_code: str) -> OpenAIConfigEntity | None:
        if not config_code:
            return None

        try:
            XCache = XCacheFactory().get_default()
            entity = XCache.get(key=f"AI:OpenAI:Config:{config_code}", cls=OpenAIConfigEntity)
            if entity:
                return entity

            with get_read_session() as session:
                entity = session.query(OpenAIConfigEntity).filter(
                    OpenAIConfigEntity.config_code == config_code,
                ).first()
                if entity:
                    # 显式脱离会话
                    session.expunge(entity)
                    XCache.set(f"AI:OpenAI:Config:{config_code}", entity)

                return entity

        except Exception as e:
            logger.error(f"[{config_code}] - 查询详情失败: {e}")

        return None
