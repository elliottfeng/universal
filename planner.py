from langchain.chains.prompt_selector import is_llm
from langchain.llms.base import LLM
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def planner(sex, age, height, weight, waistline, neckline, body_info, target, api_key):
    user_info = ChatPromptTemplate.from_messages([
        ("human",
         "性别：{sex} 年龄：{age} 身高：{height} 体重：{weight} 腰围：{waistline} 颈围：{neckline},请帮我计算出我的体脂率和基础代谢值，只展示结果。参考格式：我的体脂率为20%，我的基础代谢值为1600Kcal。")
    ])

    user_target = ChatPromptTemplate.from_messages([
        ("human",
         "{body_info},我的身体管理目标是（增肌，减脂，保持）：{target},你是一位经验丰富的健身营养师，请根据我的体脂率和基础代谢值,参考我的身材管理的目标为{target},为我定制每日的营养摄入量。")
    ])

    llm = ChatOpenAI(model="deepseek-v4-flash", api_key=api_key, base_url="https://vip.apiyi.com/v1", max_tokens=1024)

    info_chain = user_info | llm
    target_chain = user_target | llm

    # 计算体脂率和基础代谢值
    body_info = info_chain.invoke({
        "sex": sex,
        "age": age,
        "height": height,
        "weight": weight,
        "waistline": waistline,
        "neckline": neckline
    }).content

    # 根据体脂率和基础代谢值定制营养摄入量
    request = target_chain.invoke({
        "body_info": body_info,
        "target": target
    }).content

    return body_info, request


# 测试函数
#print(planner("男", 36, 175, 85, 96, 50, "body_info", "减脂", "sk-mcPCh2zIXjSTN53c23B73c9316D74e47A50eD42c52692a43"))
