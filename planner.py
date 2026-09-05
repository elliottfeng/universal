from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# 初始化主模型和备用模型
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

# 获取模型回复（带超时控制）- 使用 Chain 方式
def get_model_response_with_timeout(chain, input_data, backup_chain=None, timeout=10):
    """使用 chain.invoke 并带超时控制"""
    with ThreadPoolExecutor() as executor:
        # 提交主模型请求
        future = executor.submit(chain.invoke, input_data)
        try:
            # 设置超时时间
            response = future.result(timeout=timeout)
            return response
        except TimeoutError:
            # 如果主模型超时，切换到备用模型
            print("主模型响应超时，正在切换到备用模型...")
            if backup_chain:
                backup_future = executor.submit(backup_chain.invoke, input_data)
                try:
                    response = backup_future.result(timeout=timeout)
                    return response
                except TimeoutError:
                    raise TimeoutError("备用模型也响应超时")
            else:
                raise

# 健身计划生成函数
def planner(sex, age, height, weight, waistline, neckline, body_info, target, api_key):
    # 初始化主模型和备用模型
    main_model, backup_model = init_models(api_key)

    # 构建用户信息提示模板
    user_info = ChatPromptTemplate.from_messages([
        ("human",
         "性别：{sex} 年龄：{age} 身高：{height} 体重：{weight} 腰围：{waistline} 颈围：{neckline},请帮我计算出我的体脂率和基础代谢值，只展示结果。参考格式：我的体脂率为20%，我的基础代谢值为1600Kcal。")
    ])

    # 构建用户目标提示模板
    user_target = ChatPromptTemplate.from_messages([
        ("human",
         "{body_info},我的身体管理目标是（增肌，减脂，保持）：{target},你是一位经验丰富的健身营养师，请根据我的体脂率和基础代谢值,参考我的身材管理的目标为{target},为我定制每日的营养摄入量。")
    ])

    # 创建 Chains
    info_chain = user_info | main_model
    info_backup_chain = user_info | backup_model
    
    target_chain = user_target | main_model
    target_backup_chain = user_target | backup_model

    # 准备输入数据
    info_input = {
        "sex": sex,
        "age": age,
        "height": height,
        "weight": weight,
        "waistline": waistline,
        "neckline": neckline
    }

    # 计算体脂率和基础代谢值（带超时控制）
    try:
        body_info_response = get_model_response_with_timeout(
            info_chain, info_input, info_backup_chain, timeout=10
        )
        body_info_result = body_info_response.content
    except TimeoutError:
        print("所有模型响应超时，使用默认值或重试...")
        # 这里可以添加重试逻辑或使用默认值
        raise

    # 准备目标输入数据
    target_input = {
        "body_info": body_info_result,
        "target": target
    }

    # 根据体脂率和基础代谢值定制营养摄入量（带超时控制）
    try:
        target_response = get_model_response_with_timeout(
            target_chain, target_input, target_backup_chain, timeout=10
        )
        request = target_response.content
    except TimeoutError:
        print("所有模型响应超时，使用默认值或重试...")
        raise

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
    except TimeoutError:
        print("模型响应超时，请稍后重试")
    except Exception as e:
        print(f"发生错误: {e}")
