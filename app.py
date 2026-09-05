from langchain.chat_models import ChatOpenAI  # 改用和正常app一样的导入
from langchain.schema import SystemMessage, HumanMessage
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# 初始化主模型和备用模型（和正常app保持一致）
def init_models(api_key):
    main_model = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=api_key,
        base_url="https://vip.apiyi.com/v1",
        max_tokens=1024
    )
    backup_model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        base_url="https://vip.apiyi.com/v1",
        max_tokens=1024
    )
    return main_model, backup_model

# 获取模型回复（带超时控制）- 修改为使用 messages
def get_model_response_with_timeout(model, backup_model, prompt_text, timeout=10):
    """使用 HumanMessage 包装提示"""
    messages = [HumanMessage(content=prompt_text)]
    
    with ThreadPoolExecutor() as executor:
        # 提交主模型请求
        future = executor.submit(model, messages)  # 直接调用模型，而不是 invoke
        try:
            # 设置超时时间
            response = future.result(timeout=timeout)
            return response
        except TimeoutError:
            # 如果主模型超时，切换到备用模型
            print("主模型响应超时，正在切换到备用模型...")
            backup_future = executor.submit(backup_model, messages)
            try:
                response = backup_future.result(timeout=timeout)
                return response
            except TimeoutError:
                raise TimeoutError("备用模型也响应超时")

# 健身计划生成函数
def planner(sex, age, height, weight, waistline, neckline, body_info, target, api_key):
    # 初始化主模型和备用模型
    main_model, backup_model = init_models(api_key)

    # 构建用户信息提示（使用简单字符串，和正常app保持一致）
    body_info_prompt = f"""性别：{sex} 年龄：{age} 身高：{height} 体重：{weight} 腰围：{waistline} 颈围：{neckline}
请帮我计算出我的体脂率和基础代谢值，只展示结果。
参考格式：我的体脂率为20%，我的基础代谢值为1600Kcal。"""

    # 计算体脂率和基础代谢值（带超时控制）
    body_info_response = get_model_response_with_timeout(
        main_model, backup_model, body_info_prompt
    )
    
    # 提取响应内容（和正常app保持一致的处理方式）
    if hasattr(body_info_response, 'content'):
        body_info_result = body_info_response.content
    else:
        body_info_result = str(body_info_response)

    # 构建用户目标提示
    target_prompt = f"""{body_info_result}
我的身体管理目标是（增肌，减脂，保持）：{target}
你是一位经验丰富的健身营养师，请根据我的体脂率和基础代谢值，参考我的身材管理的目标为{target}，为我定制每日的营养摄入量。"""

    # 根据体脂率和基础代谢值定制营养摄入量（带超时控制）
    target_response = get_model_response_with_timeout(
        main_model, backup_model, target_prompt
    )
    
    if hasattr(target_response, 'content'):
        request = target_response.content
    else:
        request = str(target_response)

    return body_info_result, request

# 测试函数
if __name__ == "__main__":
    api_key = "sk-mcPCh2zIXjSTN53c23B73c9316D74e47A50eD42c52692a43"
    
    try:
        print("正在生成健身计划...")
        result = planner("男", 36, 175, 85, 96, 50, "body_info", "减脂", api_key)
        print("\n=== 体脂率和基础代谢 ===")
        print(result[0])
        print("\n=== 营养计划 ===")
        print(result[1])
    except Exception as e:
        print(f"错误: {e}")
