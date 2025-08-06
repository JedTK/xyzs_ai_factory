from sqlalchemy import Column, BigInteger, String, Integer, Text

from xyzs_py import Base, XBaseEntity


class AIAssistantEntity(Base, XBaseEntity):
    """
    AI助手实体类，表示与AI助手相关的配置信息。
    """
    code = Column(String(64), nullable=False, default="", comment="助手编码")
    name = Column(String(64), nullable=False, default="", comment="助手名称")
    role_name = Column(String(32), nullable=False, default="system", comment="角色设定，一般为：system")
    role_content = Column(String(2000), nullable=False, default="", comment="角色内容")
    default_params = Column(Text, nullable=False, comment="默认参数JSON格式")
    status = Column(Integer, nullable=False, default=1, comment="状态：0-删除，1-正常")
    config_code = Column(String(64), nullable=False, default="", comment="默认使用OpenAI配置")
    model_name = Column(String(100), nullable=False, default="", comment="默认使用模型名")
    create_time = Column(BigInteger, nullable=False, default=0, comment="创建时间戳")
    update_time = Column(BigInteger, nullable=False, default=0, comment="更新时间戳")

    def __repr__(self):
        """
        定义对象的字符串表示形式，方便调试和日志记录。

        :return: 包含助手ID和助手名称的字符串。
        """
        return f"<id={self.id}, name={self.name}>"
