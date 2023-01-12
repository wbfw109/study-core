"""ml_model_backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path
from django.urls.conf import include
from ml_model.views import view_cube, view_login, view_camera, view_dashboard, view_history, view_alert, view_websocket
from rest_framework import routers
from . import consumers

from ml_model.views import (
    view_cube,
    view_login,
    view_camera,
    view_dashboard,
    view_history,
    view_test,
)

device_camera_router = routers.DefaultRouter()
device_camera_router.register(r"cameras", view_camera.DeviceCameraViewSet, basename="camera")
device_cube_router = routers.DefaultRouter()
device_cube_router.register(r"cubes", view_cube.DeviceCubeViewSet, basename="cube")


# router = routers.DefaultRouter()
# router.register(r'users', views.UserView, 'user')

urlpatterns = [
    path("admin", admin.site.urls),
    # path('auth/login/', include(router.urls)),
    # re_path('.*', TemplateView.as_view(template_name='index.html')),
    re_path('test', view_login.users_list),
    re_path('auth/login', view_login.get_pw),
    re_path('auth/signup', view_login.register_user),
    re_path('auth/id', view_login.get_id),
    re_path('auth/pw', view_login.update_pw),
    re_path('auth/create-num', view_login.create_random_num),
    re_path('dashboard/event', view_dashboard.event_list),
    re_path('dashboard/recent', view_dashboard.recent_query),
    re_path('dashboard/device-status', view_dashboard.get_device_status),
    re_path('device/create/cube', view_cube.create_cube),
    re_path("device/", include((device_camera_router.urls, ""))),
    re_path("device/", include((device_cube_router.urls, ""))),
    re_path('history', view_history.history_list),
    re_path('alert', view_alert.event_list),
    # re_path('device/cube', view_cube.cube_list),
    # re_path('device/firmware', view_cube.get_firmware),
    # re_path("signup", view_test.SignupView.as_view()),
]
websocket_urlpatterns = [
    path('websocket/ws/websocket/', consumers.WebSocketConsumer.as_asgi()),
]
