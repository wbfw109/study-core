from rest_framework.response import Response
from rest_framework.decorators import api_view

from django.db import connection
from django.db.models import Q

from ml_model.model.base import User, UserProfile
from ml_model.model.device_cube import DeviceCube
from ml_model.model.device_camera import DeviceCamera

import logging
logger = logging.getLogger(__name__)


@api_view(['GET'])
def event_list(request):
    user_name = request.GET.get('user_name', None)
    if user_name is None:
        return Response([[], 0, False])

    cursor = connection.cursor()
    user_id = User.objects.filter(username=user_name).values('id')[0]['id']

    raw = '''
        select e.id, e.threat_event_id, e.created_timestamp, c.location, e.image
        from device_camera_event e
        left join device_camera c on e.camera_id = c.id
        where (camera_id, e.created_timestamp) in
        (
            select t.camera_id, max(t.created_timestamp) as `created_timestamp`
            from device_camera_event t inner join device_camera m on t.camera_id = m.id
                                        inner join device_cube b on m.cube_id = b.id
            where b.user_id = ''' + str(user_id) + '''
            group by t.camera_id
        ) and threat_event_id <> 1
        order by e.created_timestamp desc;
    '''

    cursor.execute(raw)
    result = list(cursor.fetchall())

    datas = []
    for data in result : 
        row = {
            'id' : data[0],
            'event_id' : data[1],
            'created_timestamp' : data[2],
            'location' : data[3],
            'images' : data[4]
        }
        datas.append(row)
    
    logger.debug('get events from device_camera_event')
    logger.debug(datas)

    status = False
    if len(datas) > 0:
        status = True

    return Response([datas, len(datas), status])


@api_view(['GET'])
def recent_query(request):
    user_name = request.GET.get('user_name', None)
    if user_name is None:
        return Response([])

    user_id = User.objects.filter(username=user_name).values('id')[0]['id']

    query_dict = UserProfile.objects.filter(user_id=user_id).values('recent_camera_query')[0]
    user_cube = DeviceCube.objects.filter(user_id=user_id).values('id', 'serial_number')

    recent_device_camera = []
    for cube in user_cube:
        query = Q(ip_address=query_dict['recent_camera_query'])
        query.add(Q(location=query_dict['recent_camera_query']), Q.OR)
        query.add(Q(cube_id=cube['id']), Q.AND)
        user_camera = DeviceCamera.objects.filter(query).values('id', 'ip_address', 'location')

        recent_device_camera += user_camera

    logger.debug('get recent search results about device camera')
    logger.debug(recent_device_camera)

    return Response(recent_device_camera)


cube_dict = {}


@api_view(['GET'])
def get_device_status(request):
    # [[cube serial, camera ip, event type, event_created_datetime, image], ...]
    # camera_list = [['a', '1.1', 1, 1, 'path'], ['a', '1.2', 2, 1, 'path'], ['a', '1.3', 2, 1, 'path']]
    camera_list = []

    if len(camera_list) == 0:
        return Response({'device_count': 0})

    serial = camera_list[0][0]

    cube_dict[serial] = []

    for camera in camera_list:
        event = camera[2]
        if event != 1:
            cube_dict[serial].append(camera)

    device_count = 0
    for cube_serial in cube_dict:
        device_count += len(cube_dict[cube_serial])

    return Response({'device_count': device_count})
