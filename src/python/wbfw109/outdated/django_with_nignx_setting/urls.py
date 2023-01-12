"""
Definition of urls for Minecraft_OOPARTS_Web.
"""

from datetime import datetime
from django.conf.urls import url
import django.contrib.auth.views

import app.forms
import app.views

# Uncomment the next lines to enable the admin:
from django.conf.urls import include
from django.contrib import admin
admin.autodiscover()

urlpatterns = [
    url(r'^$', app.views.show, name='show'),
    url(r'^home$', app.views.home, name='home'),
    url(r'^contact$', app.views.contact, name='contact'),

    url(r'^about', app.views.about, name='about'),
    url(r'^feature$', app.views.feature, name='feature'),
    url(r'^history$', app.views.history, name='history'),

    url(r'^environment$', app.views.environment, name='environment'),
    url(r'^best$', app.views.best, name='best'),

    url(r'^login/$',
        django.contrib.auth.views.LoginView.as_view(
            template_name = 'app/login.html', 
            authentication_form = app.forms.BootstrapAuthenticationForm, 
            extra_context = {       # variables like normal view.py 
                'title': '로그인',
                'year': datetime.now().year,
                }),
        name='login'),
    url(r'^logout$',
        django.contrib.auth.views.LogoutView.as_view(
            template_name = 'app/logged_off.html',
            # next_page= '/'
            ),
        name='logout'),


    # Uncomment the admin/doc line below to enable admin documentation:
    url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    
    url(r'^admin/', admin.site.urls),

]





"""
login, logout
    Deprecated since version 1.11: The login function-based view should be replaced by the class-based LoginView
url(r'^admin/', include(admin.site.urls)),
    Deprecated

    !! append Comma (,) !! on end
"""