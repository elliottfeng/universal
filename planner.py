from langchain.chains.prompt_selector import is_llm
from langchain.llms.base import LLM
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# 初始化主模型和备用模型
def init_models(api_key):
    main_model = ChatOpenAI(
        model="deepseek-chat",  # 主模型名称
        api_key=api_key,        # 主模型 API 密钥
        base_url="https://vip.apiyi.com/v1",  # 主模型 API 地址
        max_tokens=1024
    )
    backup_model = ChatOpenAI(
        model="gpt-4o-mini",    # 备用模型名称
        api_key=api_key,        # 备用模型 API 密钥
        base_url="https://vip.apiyi.com/v1",  # 备用模型 API 地址
        max_tokens=1024
    )
    return main_model, backup_model

# 获取模型回复（带超时控制）
def get_model_response_with_timeout(model, backup_model, prompt, timeout=10):
    with ThreadPoolExecutor() as executor:
        # 提交模型请求
        future = executor.submit(model.invoke, prompt)
        try:
            # 设置超时时间
            response = future.result(timeout=timeout)
            return response
        except TimeoutError:
            # 如果主模型超时，切换到备用模型
            print("主模型响应超时，正在切换到备用模型...")
            return backup_model.invoke(prompt)

# 健身计划生成函数
def planner(sex, age, height, weight, waistline, neckline, body_info, target, api_key):
    # 初始化主模型和备用模型
    main_model, backup_model = init_models(api_key)

    # 构建用户信息提示模板
    user_info = ChatPromptTemplate.from_messages([
        ("human", "性别：{sex} 年龄：{age} 身高：{height} 体重：{weight} 腰围：{waistline} 颈围：{neckline},请帮我计算出我的体脂率和基础代谢值，只展示结果。参考格式：我的体脂率为20%，我的基础代谢值为1600Kcal。")
    ])

    # 构建用户目标提示模板
    user_target = ChatPromptTemplate.from_messages([
        ("human", "{body_info},我的身体管理目标是（增肌，减脂，保持）：{target},你是一位经验丰富的健身营养师，请根据我的体脂率和基础代谢值,参考我的身材管理的目标为{target},为我定制每日的营养摄入量。")
    ])

    # 计算体脂率和基础代谢值（带超时控制）
    body_info_prompt = user_info.format_messages(
        sex=sex, age=age, height=height, weight=weight, waistline=waistline, neckline=neckline
    )
    body_info_response = get_model_response_with_timeout(main_model, backup_model, body_info_prompt)
    body_info = body_info_response.content

    # 根据体脂率和基础代谢值定制营养摄入量（带超时控制）
    target_prompt = user_target.format_messages(
        body_info=body_info, target=target
    )
    target_response = get_model_response_with_timeout(main_model, backup_model, target_prompt)
    request = target_response.content

    return body_info, request

# 测试函数
if __name__ == "__main__":
    api_key = "sk-mcPCh2zIXjSTN53c23B73c9316D74e47A50eD42c52692a43"
    result = planner("男", 36, 175, 85, 96, 50, "body_info", "减脂", api_key)
    print(result)
