from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def planner(sex, age, height, weight, waistline, neckline, body_info, target, api_key):
    # 构建用户信息提示模板（计算体脂率和基础代谢）
    user_info = ChatPromptTemplate.from_messages([
        ("human",
         "性别：{sex} 年龄：{age} 身高：{height} 体重：{weight} 腰围：{waistline} 颈围：{neckline},请帮我计算出我的体脂率和基础代谢值，只展示结果。参考格式：我的体脂率为20%，我的基础代谢值为1600Kcal。")
    ])

    # 构建用户目标提示模板（制定营养计划）
    user_target = ChatPromptTemplate.from_messages([
        ("human",
         "{body_info},我的身体管理目标是（增肌，减脂，保持）：{target},你是一位经验丰富的健身营养师，请根据我的体脂率和基础代谢值,参考我的身材管理的目标为{target},为我定制每日的营养摄入量。")
    ])

    # 初始化LLM
    llm = ChatOpenAI(
        model="deepseek-v4-flash", 
        api_key=api_key, 
        base_url="https://vip.apiyi.com/v1", 
        max_tokens=1024
    )

    # 创建处理链
    info_chain = user_info | llm
    target_chain = user_target | llm

    # 第一步：计算体脂率和基础代谢值
    # 注意：这里使用 body_info_result 避免与参数 body_info 冲突
    body_info_result = info_chain.invoke({
        "sex": sex,
        "age": age,
        "height": height,
        "weight": weight,
        "waistline": waistline,
        "neckline": neckline
    }).content

    # 第二步：根据体脂率和基础代谢值定制营养摄入量
    # 使用 body_info_result 作为输入
    request = target_chain.invoke({
        "body_info": body_info_result,  # 使用计算得到的结果
        "target": target
    }).content

    # 返回结果
    return body_info_result, request


# 测试函数
if __name__ == "__main__":
    api_key = "sk-mcPCh2zIXjSTN53c23B73c9316D74e47A50eD42c52692a43"
    
    try:
        print("正在生成健身计划...")
        body_info, plan = planner(
            sex="男", 
            age=36, 
            height=175, 
            weight=85, 
            waistline=96, 
            neckline=50, 
            body_info="body_info",  # 这个参数在代码中实际上没有被使用
            target="减脂", 
            api_key=api_key
        )
        
        print("\n=== 体脂率和基础代谢 ===")
        print(body_info)
        print("\n=== 营养计划 ===")
        print(plan)
        
    except Exception as e:
        print(f"发生错误: {e}")
