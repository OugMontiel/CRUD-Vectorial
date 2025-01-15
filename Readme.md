# CRUD con ChromaDB

Este proyecto demuestra cómo realizar operaciones CRUD (Crear, Leer, Actualizar y Eliminar) utilizando ChromaDB con la última configuración. A continuación, se detallan los pasos y configuraciones utilizados.

## 1. Métodos de consulta en el CRUD

- **Crear (Create):** Se utiliza el método `add` para insertar nuevos documentos en la colección. Este método requiere listas de `documents`, `metadatas` e `ids`.

- **Leer (Read):** Para consultar documentos similares, se emplea el método `query`, proporcionando una lista de `query_texts` y especificando `n_results` para determinar el número de resultados deseados.

- **Actualizar (Update):** ChromaDB no permite la actualización directa de documentos existentes. Para actualizar un documento, es necesario eliminarlo utilizando `delete` y luego volver a agregarlo con los cambios deseados mediante `add`.

- **Eliminar (Delete):** El método `delete` permite eliminar documentos de la colección especificando sus `ids`.

## 2. Configuración de la base de datos vectorial

Se inicializa un cliente de ChromaDB con la configuración adecuada, especificando el directorio de persistencia:

```python
client = chromadb.PersistentClient(path="/path/to/persist/directory")
```

Esto asegura que los datos se almacenen en el directorio especificado, permitiendo la persistencia entre sesiones.

## 3. Funcionamiento de los embeddings y su configuración

Los embeddings son representaciones vectoriales de texto que permiten medir similitudes semánticas entre documentos. En este proyecto, se utiliza la función de embedding `SentenceTransformerEmbeddingFunction` con el modelo `all-MiniLM-L6-v2`:

```python
from chromadb.utils import embedding_functions
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
```

Al crear la colección, se asigna esta función de embedding para que ChromaDB la utilice al agregar y consultar documentos.

## 4. Requisitos

Asegúrate de tener instaladas las dependencias necesarias, incluyendo `chromadb` y `torch`. Puedes instalarlas utilizando:

```bash
pip install chromadb toch
```

