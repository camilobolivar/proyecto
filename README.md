# Descripción del archivo 
Este archivo contiene una descripción lógica del código utilizado para implementar el algoritmo Actor-Critic en el entorno "CartPole-v0" utilizando TensorFlow y Keras.

# Importación de bibliotecas y configuración de parámetros 
En esta sección se importan las bibliotecas necesarias y se establecen los parámetros de configuración. Esto incluye la semilla para reproducibilidad, el factor de descuento gamma y el número máximo de pasos por episodio. También se crea el entorno CartPole-v0 y se establece una semilla para el entorno.

# Implementación de la red Actor-Critic 
En esta sección se implementa la red Actor-Critic utilizando TensorFlow y Keras. La red consta de tres capas: una capa de entrada con forma (num_inputs,), una capa oculta completamente conectada con activación ReLU y una capa de salida para las acciones con activación softmax, y una capa de salida para el crítico que estima las recompensas futuras. El modelo se crea utilizando la API funcional de Keras.

# Bucle principal de entrenamiento 
En esta sección comienza el bucle principal de entrenamiento. Se reinicia el estado del entorno y se inicializan las variables para el historial de probabilidades de acción, valores del crítico y recompensas. Luego, se ejecutan los pasos del episodio dentro de un contexto GradientTape de TensorFlow para realizar el cálculo automático de gradientes.

Dentro del bucle, se predecen las probabilidades de acción y el valor del crítico para el estado actual del entorno. Se registran estos valores en los historiales correspondientes. Luego, se muestrea una acción de acuerdo con las probabilidades de acción predichas y se aplica la acción al entorno. Se registran la recompensa y la recompensa acumulada del episodio.

El bucle continúa hasta que se alcanza el número máximo de pasos por episodio o hasta que el entorno reporta que el episodio ha terminado (done=True).

Después de ejecutar los pasos del episodio, se actualiza la recompensa acumulada promediada en running_reward. A continuación, se calculan los valores esperados de las recompensas utilizando el factor de descuento gamma. Estos valores se utilizan como etiquetas para el crítico. Luego, los historiales se normalizan y se calculan las pérdidas del actor y el crítico.

Después del cálculo de las pérdidas, se realiza la retropropagación y la actualización de los parámetros del modelo utilizando el optimizador Adam. Luego, se borran los historiales de probabilidades de acción, valores del crítico y recompensas para el siguiente episodio.

Se registran los detalles del episodio, como la recompensa acumulada promediada, y se verifica si se ha alcanzado la condición para considerar que se ha resuelto la tarea (recompensa promediada mayor a 195). Si se cumple la condición, se imprime un mensaje y se rompe el bucle de entrenamiento.

En resumen, este código implementa el algoritmo Actor-Critic para entrenar un modelo que resuelva el entorno "CartPole-v0" utilizando TensorFlow y Keras.
