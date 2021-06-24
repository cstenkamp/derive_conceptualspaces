mkdir data
git clone https://github.com/cstenkamp/derive_conceptualspaces.git
cd derive_conceptualspaces
cp docker/sample.env docker/.env
set /p DUMMY="After you click enter, a notepad opens with a file where you have to enter your seafile-credentials. Please enter them, then come back to this window and press any key."
notepad docker/.env
pause
docker build -f Dockerfile --rm --tag derive_conceptualspaces .
FOR /F "tokens=* USEBACKQ" %%F IN (realpath ../data) DO (
SET data_path=%%F
)
docker run -it --name derive_conceptualspaces_cont -v %data_path%:/opt/data --env-file ./docker/.env  derive_conceptualspaces zsh