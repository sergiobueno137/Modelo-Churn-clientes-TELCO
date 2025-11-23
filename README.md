Ejecución de Tests Funcionales del Modelo churn clientes TELCO

Paso 1: Configurar git
Abra una Terminal en JupyterLab desde Anaconda Promt e ingrese los siguientes comandos

git config --global user.name "<USER>"
git config --global user.email <CORREO>

Paso 2: Clonar el Proyecto desde su propio Github
git clone https://github.com/<USER>/Modelo-Churn-clientes-TELCO
Paso 5: Instalar los pre-requisitos
cd Modelo-Churn-clientes-TELCO/

Paso 3: Ejecutar las pruebas en el entorno
cd src

# python make_dataset.py // omitimos este paso

python train.py

python evaluate.py

python predict.py

cd ..
Paso 4: Guardar los cambios en el Repo
git add .

git commit -m "Pruebas Finalizadas"

git push
