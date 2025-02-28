from dashscope import Generation
from datetime import datetime
import random
import json
import requests
 
# 定义工具列表，模型在选择使用哪个工具时会参考工具的name和description
tools = [
    # 工具1 获取当前时刻的时间
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {}  # 因为获取当前时间无需输入参数，因此parameters为空字典
        }
    },  
    # 工具2 获取指定城市的天气
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {  # 查询天气时需要提供位置，因此参数设置为location
                        "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                    }
                }
            },
            "required": [
                "location"
            ]
        }
    }
]
 
# 基于外部网站的天气查询工具。返回结果示例：{"location": "\u5317\u4eac\u5e02", "weather": "clear sky", "temperature": 17.94}
def get_current_weather(location):
    api_key = "9fa42a532a9338bf217afd7ac47a565a"  # 替换为你自己的OpenWeatherMap API密钥，用我的也无所谓啦，反正免费。
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return json.dumps({"location": location, "weather": weather, "temperature": temp})
    else:
        return json.dumps({"location": location, "error": "Unable to fetch weather data"})
 
 
 
# 查询当前时间的工具。返回结果示例：“当前时间：2024-04-15 17:15:18。“
def get_current_time():
    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 格式化当前日期和时间
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # 返回格式化后的当前时间
    return f"当前时间：{formatted_time}。"
 
# 封装模型响应函数
def get_response(messages):
    response = Generation.call(
        model='qwen-plus',
        messages=messages,
        tools=tools,
        seed=random.randint(1, 10000),  # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
        result_format='message'  # 将输出设置为message形式
    )
    return response
 
def call_with_messages():
    print('\n')
    messages = [
            {
                "content": input('请输入：'),  # 提问示例："现在几点了？" "一个小时后几点" "北京天气如何？"
                "role": "user"
            }
    ]
    
    # 模型的第一轮调用
    first_response = get_response(messages)
    assistant_output = first_response.output.choices[0].message
    print(f"\n大模型第一轮输出信息：{first_response}\n")
    messages.append(assistant_output)
    if 'tool_calls' not in assistant_output:  # 如果模型判断无需调用工具，则将assistant的回复直接打印出来，无需进行模型的第二轮调用
        print(f"最终答案：{assistant_output.content}")
        return
    # 如果模型选择的工具是get_current_weather
    elif assistant_output.tool_calls[0]['function']['name'] == 'get_current_weather':
        tool_info = {"name": "get_current_weather", "role":"tool"}
        location = json.loads(assistant_output.tool_calls[0]['function']['arguments'])['properties']['location']['description']        
        # print(location)
        tool_info['content'] = get_current_weather(location)
    # 如果模型选择的工具是get_current_time
    elif assistant_output.tool_calls[0]['function']['name'] == 'get_current_time':
        tool_info = {"name": "get_current_time", "role":"tool"}
        tool_info['content'] = get_current_time()
    print(f"工具输出信息：{tool_info['content']}\n")
    messages.append(tool_info)
 
    # 模型的第二轮调用，对工具的输出进行总结
    second_response = get_response(messages)
    print(f"大模型第二轮输出信息：{second_response}\n")
    print(f"最终答案：{second_response.output.choices[0].message['content']}")
 
if __name__ == '__main__':
    call_with_messages()