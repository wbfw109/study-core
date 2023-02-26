from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.db import connection
import logging
from ml_model.model.base import User


logger = logging.getLogger(__name__)


@api_view(['GET'])
def history_list(request):
    user_name = request.GET.get('user_name', None)
    if user_name is None:
        return Response({'status': False, 'total': 0, 'list': []})
    user_id = User.objects.filter(username=user_name).values('id')[0]['id']
    cursor = connection.cursor()
    
    raw = '''
    SELECT e.id, e.image, DATE_FORMAT(e.created_timestamp, '%Y.%m.%d %H:%i:%s'), e.camera_id,t.id, c.location, b.serial_number, b.user_id
    FROM device_camera_event e, threat_event t, device_camera c, device_cube b 
    WHERE e.threat_event_id = t.id AND e.camera_id = c.id AND c.cube_id = b.id and (t.id<>1 and t.id<>4) and b.user_id = ''' + str(user_id) + '''
    order by e.created_timestamp desc;
    '''

    cursor.execute(raw)
    result = list(cursor.fetchall())

    datas = []
    for data in result : 
        row = {
            'id' : data[0],
            'image' : data[1],
            'created_timestamp' : data[2],
            'camera_id' : data[3],
            'event_id' : data[4],
            'location' : data[5],
            'serial_number' : data[6],
            'user_id' : data[7]
        }
        datas.append(row)
    select_datas = []    
    if 'startDate' and 'endDate' in request.GET :
        startDate = request.GET['startDate']
        endDate = request.GET['endDate']
        
        for data in datas : 
            if data['created_timestamp'] >= startDate and data['created_timestamp'] < endDate: 
                select_datas.append(data)
        logger.debug('get select events from device_camera_event')
        logger.debug(select_datas)
        return Response([select_datas, len(select_datas)])
    else : 
        logger.debug('get events from device_camera_event')
        logger.debug(datas)
        return Response([datas, len(datas)])