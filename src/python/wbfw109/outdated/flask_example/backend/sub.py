# import django
# django.setup()
# import paho.mqtt.client as mqtt
# import json, datetime
# import logging
# from ml_model.model.base import ThreatEvent
# from ml_model.model.device_cube import DeviceCube
# from ml_model.model.device_camera import DeviceCamera, DeviceCameraEvent
# from ml_model.model.base import ThreatEvent
# import websocket

# ARRAY_FIRST = 0

# logger = logging.getLogger(__name__)

# def bubble_sort(array):
#     temp = []
#     for i in range(len(array)) :
#         for j in range (len(array)) :
#             j = j + i + 1
#             if j is (len(array)) :
#                 break
#             if array [i]['event_created_datetime'] > array [j]['event_created_datetime'] :
#                 temp = array[i]
#                 array[i]=array[j]
#                 array[j]=temp
        
        
        
# def on_message(client, userdata, msg):
#     str_data =msg.payload.decode('utf-8')
#     event_lists = json.loads(str_data)
#     current_time = datetime.datetime.now()
#     cube_info = DeviceCube.objects.filter(serial_number=event_lists[0]['cube_serial'])
    
#     # 데이터 날짜순으로 정렬
#     bubble_sort(event_lists)

#     # 큐브에서 들어온 데이터 저장
#     for event in event_lists:
#         camera_info = DeviceCamera.objects.get(id=cube_info[ARRAY_FIRST].id)
#         threat_type = ThreatEvent.objects.get(id=event['event_type'])
        
#         event_save = DeviceCameraEvent(camera=camera_info,
#                                     threat_event=threat_type, 
#                                     images=event['images'], 
#                                     created_datetime=event['event_created_datetime'])
#         event_save.save()
        
#     mqtt_lists={}
#     alret_lists=[]
    
#     # event_alret
#     if len(event_lists) == 1 and event_lists[ARRAY_FIRST]['event_type'] == '1':
#         pass
#     else :
#         for event in event_lists :
#             if (event['event_type'] == '1' or event['event_type'] == '4') :
#                 continue
#             time_result = ''
#             time = (current_time - datetime.datetime.strptime(event['event_created_datetime'], "%Y-%m-%d %H:%M:%S.%f")).seconds
#             hours = int(time / 3600)
#             minutes = int((time % 3600) / 60)
            
#             if hours > 0 :
#                 time_result = str(hours) + '시간 전'
#             elif hours == 0 and minutes > 0 :
#                 time_result = str(minutes) + '분 전'
#             else :
#                 time_result = '조금 전'
                
#             event_name = ThreatEvent.objects.get(id=event['event_type'])
#             data = (event_name.name, time_result, event['event_type'], event['cube_serial'])
            
#             alret_lists.append(data)
#         alret_lists.reverse()    
        
#         mqtt_lists['alret_lists']=alret_lists
#         send_data = json.dumps(mqtt_lists,separators=(',', ':'), ensure_ascii=False)
#         ws = websocket.WebSocket()
#         ws.connect("ws://127.0.0.1:8000/websocket/ws/websocket/")
#         ws.send(send_data)
#         ws.close()


# # 새로운 클라이언트 생성
# client = mqtt.Client()

# # 콜백 함수 설정 on_connect(브로커에 접속), on_disconnect(브로커에 접속중료), on_subscribe(topic 구독),
# # on_message(발행된 메세지가 들어왔을 때)
# client.on_message = on_message
# # address : localhost, port: 1883 에 연결
# client.connect('localhost', 1883)
# # common topic 으로 메세지 발행
# client.subscribe('event', 1)
