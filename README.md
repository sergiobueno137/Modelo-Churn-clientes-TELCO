
# Ejecución de Tests Funcionales del Modelo churn clientes TELCO

### Paso 1: Configurar git

Abra una Terminal en JupyterLab e ingrese los siguientes comandos

```
git config --global user.name "<USER>"
git config --global user.email <CORREO>
```


### Paso 2: Clonar el Proyecto desde su propio Github

```
git clone https://github.com/<USER>/Modelo-Churn-clientes-TELCO.git
```


### Paso 3: Instalar los pre-requisitos

```
cd Modelo-Churn-clientes-TELCO/

```


### Paso 4: Ejecutar las pruebas en el entorno

```
cd src

python train.py

python evaluate.py

python predict.py

cd ..
```


### Paso 5: Guardar los cambios en el Repo

```
git add .

git commit -m "Pruebas Finalizadas"

git push

```

Ingrese su usuario y Personal Access Token de Github. Puede revisar que los cambios se hayan guardado en el repositorio. Luego, puede finalizar JupyterLab ("File" => "Shut Down").
