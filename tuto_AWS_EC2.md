dans l'ordre : 
- tu push le code de ton app sur un repo github
- tu crées un compte aws
- tu crées une instance EC2 (moi j'ai choisi un Ubuntu Server, un Debian peut faire l'affaire aussi)
- tu augmentes l'espace de stockage de l'instance (très important!) à 25Go
- Tu te connectes à ton instance via l'interface aws ou via SSH (la première option est la plus simple)

Dans la console du serveur:
```bash
sudo apt install -y python3-pip nginx
```

Ensuite on utilise nginx pour rerouter 127.0.0.1 vers le web. Pour faire ça, on crée un fichier de config pour nginx :
```bash
sudo vim /etc/nginx/sites-enabled/fastapi_nginx
```
dans le fichier fastapi_nginx:
```bash
server {
        listen 80;
        server_name XX.XX.XX.XXX; # IPV4 publique de ton instance EC2
        location / {
                proxy_pass http://12.0.0.1:8000;
        }
}
```

retour dans la console :
```bash
sudo service nginx restart
```

```bash
git clone https://github.com/... # l'url de ton repo github pour l'app
```

```bash
cd leDossierDeTonProjet
pip3 install -r requirements.txt
```

```bash
python3 uvicorn main:app
```

Et BAM!!! :sunglasses:
Maintenant dans ton navigateur, quand tu te rends à l'ipv4 publique de ton instance EC2, t'as ton app qui tourne comme un derviche !!! et plus important, quand tu quittes la console, ça tourne toujours

NB : tu peux avoir des soucis lors de l'install de tensorflow. mais il y a un workaround :
- installation manuelle mais en remplaçant tensorflow par tensorflow-cpu :
```bash
pip install tensorflow-cpu --no-cache-dir
```

installer les packages suivants peut aussi aider :
```bash
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
```