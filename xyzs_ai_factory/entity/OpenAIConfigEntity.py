from sqlalchemy import Column, BigInteger, String, Integer

from xyzs_py import Base, XBaseEntity


class OpenAIConfigEntity(Base, XBaseEntity):
    """
    OpenAI配置实体类，表示与OpenAI API相关的配置数据。
    """
    config_code = Column(String(64), nullable=False, default="", comment="配置编码")
    config_name = Column(String(64), nullable=False, default="", comment="配置名称")
    remark = Column(String(255), nullable=False, default="", comment="备注")
    base_url = Column(String(200), nullable=False, default="", comment="API基础Url")
    api_key = Column(String(100), nullable=False, default="", comment="API秘钥")
    type_code = Column(String(100), nullable=False, default="OpenAI", comment="类型")
    connect_timeout = Column(Integer, nullable=False, default=60, comment="HTTP请求连接超时时间，秒")
    read_timeout = Column(Integer, nullable=False, default=60, comment="HTTP读取超时时间，秒")
    write_timeout = Column(Integer, nullable=False, default=60, comment="HTTP写入超时时间，秒")
    status = Column(Integer, nullable=False, default=1, comment="状态：0-删除，1-正常")
    create_time = Column(BigInteger, nullable=False, default=0, comment="创建时间戳")

    def __repr__(self):
        """
        定义对象的字符串表示形式，方便调试和日志记录。

        :return: 包含配置ID和配置编码的字符串。
        """
        return f"<id={self.id}, config_code={self.config_code}>"
