## How to install Docker

To run this code with minimal overhead of installing (and without the possibility to alter code), the recommended way is Docker. The following guides you through the process of setting up `git`, `docker` and `docker-compose`.

### Linux
Instructions tested on Ubuntu 20.04. Just executing the following block should be enough:

```
sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release -y
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt-get install docker-ce docker-ce-cli containerd.io -y
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
sudo apt-get install git -y
```

  
### Windows

* Install Docker Desktop for Windows (https://docs.docker.com/docker-for-windows/install). Installer is >500mb so quite big, and installation requires a restart (and afterwards it prompted me to install https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi ), but just follow the instructions of the installer.
* Install Git for Windows: Download the `.exe` from https://git-scm.com/download/win and run the installer. In the install wizard, make sure that git can be used from the command prompt, otherwise you'd have to switch between shells when coding and committing to git. Further use one of the two commit unix style options. Other than that, you'll probably go for the openSSL as well as Windows default console as terminal emulator options.