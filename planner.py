from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# 初始化主模型和备用模型
def init_models(api_key):
    main_model = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=api_key,
        base_url="https://vip.apiyi.com/v1",
        max_tokens=1024,
        timeout=30  # 添加超时参数
    )
    backup_model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        base_url="https://vip.apiyi.com/v1",
        max_tokens=1024,
        timeout=30
    )
    return main_model, backup_model

# 获取模型回复（带超时控制）
def get_model_response_with_timeout(model, backup_model, prompt, timeout=10):
    with ThreadPoolExecutor() as executor:
        future = executor.submit(model.invoke, prompt)
        try:
            response = future.result(timeout=timeout)
            return response
        except TimeoutError:
            print("主模型响应超时，正在切换到备用模型...")
            # 备用模型也添加超时控制
            backup_future = executor.submit(backup_model.invoke, prompt)
            try:
                response = backup_future.result(timeout=timeout)
                return response
            except TimeoutError:
                raise TimeoutError("备用模型也响应超时")

# 健身计划生成函数
def planner(sex, age, height, weight, waistline, neckline, user_body_info, target, api_key):
    # 初始化主模型和备用模型
    main_model, backup_model = init_models(api_key)

    # 修改为使用简单的字符串模板，避免 ChatPromptTemplate 的问题
    body_info_template = """性别：{sex} 年龄：{age} 身高：{height} 体重：{weight} 腰围：{waistline} 颈围：{neckline}
请帮我计算出我的体脂率和基础代谢值，只展示结果。
参考格式：我的体脂率为20%，我的基础代谢值为1600Kcal。"""
    
    target_template = """{body_info}
我的身体管理目标是（增肌，减脂，保持）：{target}
你是一位经验丰富的健身营养师，请根据我的体脂率和基础代谢值，参考我的身材管理的目标为{target}，为我定制每日的营养摄入量。"""

    # 计算体脂率和基础代谢值（带超时控制）
    body_info_prompt = body_info_template.format(
        sex=sex, age=age, height=height, weight=weight, 
        waistline=waistline, neckline=neckline
    )
    
    body_info_response = get_model_response_with_timeout(
        main_model, backup_model, body_info_prompt
    )
    
    # 注意：这里从响应中提取内容
    if hasattr(body_info_response, 'content'):
        calculated_body_info = body_info_response.content
    else:
        calculated_body_info = str(body_info_response)

    # 根据体脂率和基础代谢值定制营养摄入量
    target_prompt = target_template.format(
        body_info=calculated_body_info, 
        target=target
    )
    
    target_response = get_model_response_with_timeout(
        main_model, backup_model, target_prompt
    )
    
    if hasattr(target_response, 'content'):
        request = target_response.content
    else:
        request = str(target_response)

    return calculated_body_info, request

# 测试函数
if __name__ == "__main__":
    api_key = "sk-mcPCh2zIXjSTN53c23B73c9316D74e47A50eD42c52692a43"
    result = planner("男", 36, 175, 85, 96, 50, "body_info", "减脂", api_key)
    print("体脂率和基础代谢：", result[0])
    print("营养计划：", result[1])
