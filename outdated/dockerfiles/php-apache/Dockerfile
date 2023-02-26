FROM php:8.0.9-apache
RUN a2enmod rewrite
RUN apt-get update && apt-get upgrade -y libpq-dev
RUN docker-php-ext-install pdo_pgsql pgsql mysqli && docker-php-ext-enable pdo_pgsql pgsql mysqli


ENV PORT 80
ENTRYPOINT []
CMD sed -si "s/^Listen [0-9]\+$/Listen $PORT/1" /etc/apache2/ports.conf && sed -s "s/^<VirtualHost \*:[0-9]\+>$/<VirtualHost *:$PORT>/1" /etc/apache2/sites-available/000-default.conf && docker-php-entrypoint apache2-foreground

# Reference
# https://github.com/docker-library/php/issues/94


