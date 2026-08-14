"""
J.A.R.V.I.S 自定义技能注册表
支持简单的命令匹配与工具调用，例如：打开软件、查询天气（示例）、系统信息等。
"""
import logging
import os
import platform
import subprocess
import webbrowser
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class SkillRegistry:
    """
    技能注册表。
    每个技能是一个函数，接收用户文本，返回执行结果或 None（不匹配）。
    """

    def __init__(self):
        self._skills: Dict[str, Callable[[str], Optional[str]]] = {}
        self._register_default_skills()

    def register(self, name: str, handler: Callable[[str], Optional[str]]):
        """注册技能。"""
        self._skills[name] = handler

    def execute(self, text: str) -> Optional[str]:
        """
        依次执行所有技能，返回第一个匹配结果。
        """
        text = text.lower().strip()
        for name, handler in self._skills.items():
            try:
                result = handler(text)
                if result:
                    logger.info("技能 %s 匹配并执行", name)
                    return result
            except Exception as e:
                logger.error("技能 %s 执行失败: %s", name, e)
        return None

    def _register_default_skills(self):
        """注册默认技能。"""
        self.register("open_browser", self._open_browser)
        self.register("open_calculator", self._open_calculator)
        self.register("system_info", self._system_info)
        self.register("weather_mock", self._weather_mock)
        self.register("time_now", self._time_now)

    # ---------- 默认技能实现 ----------

    def _open_browser(self, text: str) -> Optional[str]:
        """打开浏览器或指定网站。"""
        triggers = ["打开浏览器", "打开网页", "打开网站", "open browser", "open website"]
        if not any(t in text for t in triggers):
            return None

        if "百度" in text:
            webbrowser.open("https://www.baidu.com")
            return "已为您打开百度。"
        if "github" in text or "git" in text:
            webbrowser.open("https://github.com")
            return "已为您打开 GitHub。"
        webbrowser.open("https://www.google.com")
        return "已为您打开浏览器。"

    def _open_calculator(self, text: str) -> Optional[str]:
        """打开计算器。"""
        triggers = ["打开计算器", "open calculator"]
        if not any(t in text for t in triggers):
            return None

        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.Popen(["open", "-a", "Calculator"])
            elif system == "Windows":
                subprocess.Popen(["calc"])
            else:
                subprocess.Popen(["gnome-calculator"])
            return "计算器已打开。"
        except Exception as e:
            return f"打开计算器失败：{e}"

    def _system_info(self, text: str) -> Optional[str]:
        """查询系统信息。"""
        triggers = ["系统信息", "电脑信息", "system info", "os info"]
        if not any(t in text for t in triggers):
            return None

        return (
            f"当前操作系统：{platform.system()} {platform.release()}\n"
            f"处理器架构：{platform.machine()}\n"
            f"主机名：{platform.node()}"
        )

    def _weather_mock(self, text: str) -> Optional[str]:
        """天气查询（示例，实际可接入天气 API）。"""
        triggers = ["天气", "weather", "气温"]
        if not any(t in text for t in triggers):
            return None

        # 实际项目中可调用天气 API，这里返回示例数据
        return (
            "天气查询功能目前为演示模式。\n"
            "实际部署时可在 backend/tools/skills.py 中接入和风天气、OpenWeatherMap 等 API。"
        )

    def _time_now(self, text: str) -> Optional[str]:
        """查询当前时间。"""
        triggers = ["几点", "时间", "现在时间", "what time", "current time"]
        if not any(t in text for t in triggers):
            return None

        from datetime import datetime

        now = datetime.now()
        return f"现在是 {now.strftime('%Y年%m月%d日 %H:%M:%S')}。"


# 全局实例
SKILLS = SkillRegistry()
