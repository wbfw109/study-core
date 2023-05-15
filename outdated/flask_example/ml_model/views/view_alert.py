from rest_framework.response import Response
from rest_framework.decorators import api_view

from django.db import connection
import logging
import datetime
logger = logging.getLogger(__name__)

EVENT_TYPE_NAME = 0
EVENT_TIME = 1
EVENT_TYPE = 2


@api_view(['GET'])
def event_list(request):
    current_time = datetime.datetime.now()
    cursor = connection.cursor()
    user_id = 1
    event_lists=[]

    raw = '''
        SELECT t.name, e.created_datetime, e.threat_event_id, e.camera_id, c.location, b.serial_number, e.images
            FROM device_camera_event e, threat_event t, device_camera c, device_cube b 
            WHERE e.threat_event_id = t.id AND e.camera_id = c.id AND c.cube_id = b.id and b.user_id = ''' + str(user_id) + '''  and (t.id <> 1 and t.id<>4)
            order by e.created_datetime desc limit 10;
    '''

    cursor.execute(raw)
    result = list(cursor.fetchall())

    for data in result:
        time_result = ''
        time = (current_time - data[EVENT_TIME]).seconds
        hours = int(time / 3600)
        minutes = int((time % 3600) / 60)
        
        if hours > 0 :
            time_result = str(hours) + '시간 전'
        elif hours == 0 and minutes > 0 :
            time_result = str(minutes) + '분 전'
        else :
            time_result = '조금 전'
            
        event = (
            data[EVENT_TYPE_NAME], time_result, data[EVENT_TYPE]
        )
        event_lists.append(event)

    status = False
    if len(event_lists) > 0:
        status = True

    return Response({'status': status, 'total': len(event_lists), 'list': event_lists})