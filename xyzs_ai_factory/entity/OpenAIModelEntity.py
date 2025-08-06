from sqlalchemy import Column, BigInteger, String, Integer
from xyzs_py import Base, XBaseEntity


class OpenAIModelEntity(Base, XBaseEntity):
    """
    OpenAI模型实体类，表示与OpenAI模型相关的配置数据。
    """
    config_code = Column(String(64), nullable=False, default="", comment="配置编码")
    model_name = Column(String(100), nullable=False, default="", comment="模型名")
    is_default = Column(Integer, nullable=False, default=0, comment="默认模型：0-否，1-是")
    create_time = Column(BigInteger, nullable=False, default=0, comment="创建时间戳")

    def __repr__(self):
        """
        定义对象的字符串表示形式，方便调试和日志记录。

        :return: 包含模型ID和模型名的字符串。
        """
        return f"<id={self.id}, model_name={self.model_name}>"
