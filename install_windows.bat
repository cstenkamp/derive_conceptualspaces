mkdir data
set /p DUMMY="Now, please put your gcloud-credentials file under the name `gcloud_tools_key.json` into the `data`-directory which was just created, then come back here and press enter."
git clone https://github.com/cstenkamp/derive_conceptualspaces.git
cd derive_conceptualspaces
cp docker/sample.env docker/.env
set /p DUMMY="After you click enter, a notepad opens with a file where you have to enter your seafile-credentials. Please enter them, then close that notepad and come back to this window and press any key."
notepad docker/.env
pause
docker build -f Dockerfile --rm --tag derive_conceptualspaces .
FOR /F "tokens=* USEBACKQ" %%F IN (realpath ../data) DO (
SET data_path=%%F
)
docker run -it --name derive_conceptualspaces_cont -v %data_path%:/opt/data --env-file ./docker/.env  derive_conceptualspaces zsh