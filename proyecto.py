#Configuración
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Parámetros de configuración para toda la instalación

seed = 42
'''Establece una semilla para la generación de números aleatorios. Esto garantiza que los resultados sean reproducibles.'''
gamma = 0.99   
'''Factor de descuento para recompensas pasadas'''
max_steps_per_episode = 10000
'''Define el número máximo de pasos permitidos en un episodio del juego CartPole. Si el juego continúa durante más de este número de pasos, se considera que el episodio ha terminado.'''
env = gym.make("CartPole-v0")  
'''Crea el entorno del juego CartPole utilizando la función make de Gym. "CartPole-v0" es una cadena que identifica el entorno específico de CartPole en Gym.'''

env.seed(seed)
''' Establece la semilla del entorno del juego CartPole para que los resultados sean reproducibles.'''
eps = np.finfo(np.float32).eps.item()  
'''Define eps como el número más pequeño de punto flotante representable para el tipo np.float32. Este valor se utilizará más adelante para evitar divisiones por cero en cálculos numéricos.'''


# En esta sección se importan las bibliotecas necesarias
# y se establecen los parámetros de configuración, como 
# la semilla para reproducibilidad, el factor de descuento 
# gamma y el número máximo de pasos por episodio. También 
# se crea el entorno CartPole-v0 y se establece una semilla para el entorno.



#Implementar red de Críticos de Actores

num_inputs = 4
'''Se define el número de entradas para la red neuronal, que corresponde al estado del entorno del juego CartPole. En este caso, el estado está compuesto por 4 valores.'''
num_actions = 2
'''Se define el número de acciones posibles en el entorno del juego CartPole. En este caso, hay dos acciones posibles: mover el carro hacia la izquierda o hacia la derecha.'''
num_hidden = 128
'''Se define el número de neuronas en la capa oculta de la red neuronal. En este caso, se utiliza una capa oculta con 128 neuronas'''

inputs = layers.Input(shape=(num_inputs,))
'''Se crea una capa de entrada para el modelo de la red neuronal con la forma (num_inputs,), que corresponde al número de entradas definido anteriormente.'''
common = layers.Dense(num_hidden, activation="relu")(inputs)
'''Se crea una capa densa (totalmente conectada) con num_hidden neuronas y función de activación ReLU. Esta capa se conecta a la capa de entrada y es compartida por el Actor y el Crítico.'''
action = layers.Dense(num_actions, activation="softmax")(common)
'''Se crea una capa densa con num_actions neuronas y función de activación softmax. Esta capa representa la salida del Actor y devuelve una probabilidad para cada acción en el espacio de acciones. La función softmax asegura que las probabilidades sumen 1.'''
critic = layers.Dense(1)(common)
'''Se crea una capa densa con una sola neurona. Esta capa representa la salida del Crítico y estima las recompensas totales en el futuro. No se aplica ninguna función de activación en esta capa.'''

model = keras.Model(inputs=inputs, outputs=[action, critic])
'''Se crea el modelo de la red neuronal utilizando el objeto Model de Keras. Se especifican las capas de entrada y salida del modelo. En este caso, las salidas son la capa de acción y la capa del crítico.'''


#En esta sección se implementa la red Actor-Critic. 
# La red consta de tres capas: una capa de entrada con 
# forma (num_inputs,), una capa oculta completamente 
# conectada con activación ReLU y una capa de salida para 
# las acciones con activación softmax, y una capa de salida 
# para el crítico que estima las recompensas futuras. 
# El modelo se crea utilizando la API funcional de Keras.


# Train

optimizer = keras.optimizers.Adam(learning_rate=0.01)
'''Se crea un optimizador Adam con una tasa de aprendizaje de 0.01. El optimizador se utilizará para actualizar los pesos del modelo durante el entrenamiento.'''
huber_loss = keras.losses.Huber()
'''Se crea una función de pérdida Huber. Esta función de pérdida se utilizará para calcular la pérdida del crítico durante el entrenamiento. '''
action_probs_history = []
'''Se crea una lista para almacenar el historial de probabilidades de acción.'''
critic_value_history = []
'''Se crea una lista para almacenar el historial de valores del crítico. '''
rewards_history = []
''' Se crea una lista para almacenar el historial de recompensas. '''
running_reward = 0
'''Se inicializa el valor de running_reward en 0. Este valor se utiliza para realizar un seguimiento del promedio ponderado de las recompensas a lo largo del tiempo y verificar si el problema se ha resuelto. '''
episode_count = 0
'''Se inicializa el contador de episodios en 0. '''

while True:  # que problema hay que resolver??
    '''Se inicia un bucle que se ejecutará hasta que el problema se considere resuelto.''' 
    state = env.reset() 
    '''Se restablece el estado del entorno del juego CartPole al comienzo de cada episodio.'''
    episode_reward = 0
    '''Se inicializa la recompensa acumulada para el episodio actual en 0.'''
    with tf.GradientTape() as tape:
         '''Se inicia una cinta de gradiente para registrar las operaciones realizadas en el contexto. Esto permitirá calcular los gradientes de las variables entrenables con respecto a la pérdida posteriormente.'''
        for timestep in range(1, max_steps_per_episode):
            ''' Se inicia un bucle para ejecutar pasos en el entorno del juego hasta alcanzar el límite máximo de pasos por episodio. '''
            # env.render(); Agregar esta línea mostraría los intentos
            # del agente en una ventana emergente.

            state = tf.convert_to_tensor(state) # que es convertir en un tensorflow
            state = tf.expand_dims(state, 0) # para que coincidan con las dimensiones del modelo

            # Predecir probabilidades de acción y valores estimados de recompensa futura
            # a partir del estado del entorno
            action_probs, critic_value = model(state) # que son los valoes criticos??
            ''' Se pasa el estado a través del modelo para obtener las probabilidades de acción y el valor del crítico. '''
            critic_value_history.append(critic_value[0, 0])
            '''Se agrega el valor del crítico al historial de valores del crítico.'''

            # Muestrear acción de la distribución de probabilidad de acción
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            '''Se muestrea una acción del espacio de acciones utilizando las probabilidades de acción.'''
            action_probs_history.append(tf.math.log(action_probs[0, action]))
            '''Se agrega el logaritmo de la probabilidad de la acción seleccionada al historial de probabilidades de acción.'''

            # Aplicar la acción muestreada en nuestro entorno
            state, reward, done, _ = env.step(action)
            '''Se aplica la acción seleccionada en el entorno del juego y se obtiene el próximo estado, la recompensa, si el episodio ha terminado y otra información.'''
            rewards_history.append(reward)
            '''Se agrega la recompensa obtenida en el paso actual al historial de recompensas.'''
            episode_reward += reward
            '''Se suma la recompensa obtenida en el paso actual a la recompensa acumulada del episodio.'''

            if done:
                break




# En esta sección comienza el bucle principal de entrenamiento. 
# Se reinicia el estado del entorno y se inicializan las variables
# para el historial de probabilidades de acción, valores del crítico 
# y recompensas. Luego, se ejecutan los pasos del episodio dentro de un 
# contexto GradientTape de TensorFlow para realizar el cálculo automático de gradientes.

# Dentro del bucle, se predecen las probabilidades de acción y el valor
# del crítico para el estado actual del entorno. Se registran estos valores 
# en los historiales correspondientes. Luego, se muestrea una acción de 
# acuerdo con las probabilidades de acción predichas y se aplica la acción 
# al entorno. Se registran la recompensa y la recompensa acumulada del episodio.

# El bucle continúa hasta que se alcanza el número máximo de pasos por episodio
# o hasta que el entorno reporta que el episodio ha terminado (done=True).


        #Actualice la recompensa en ejecución para verificar la condición para resolver
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        '''Se actualiza el valor de running_reward utilizando un promedio ponderado entre el running_reward anterior y la recompensa acumulada del episodio actual. Esto se utiliza para realizar un seguimiento del rendimiento del agente a lo largo del tiempo.'''

        returns = []
        discounted_sum = 0
        ''' Se inicializa la suma descontada en 0.'''
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            '''Se calcula la suma descontada aplicando el factor de descuento gamma a la recompensa actual y sumándola a la suma descontada anterior.'''
            returns.insert(0, discounted_sum)
            '''' Se inserta el valor de la suma descontada al principio de la lista returns.'''

        # Normalizar
        returns = np.array(returns)
        '''Se convierte la lista returns en un array de NumPy.'''
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        '''Se normalizan los valores de returns restando la media y dividiendo por la desviación estándar. Esto es para estabilizar el entrenamiento.'''
        returns = returns.tolist()
        '''Se convierte el array de NumPy en una lista de Python'''

        # Cálculo de valores de pérdida para actualizar nuestra red
        history = zip(action_probs_history, critic_value_history, returns)
        '''Se crea un iterador zip que combina los historiales de probabilidades de acción, valores del crítico y valores de retorno.'''
        actor_losses = []
        '''Se crea una lista para almacenar las pérdidas del actor.'''
        critic_losses = []
        '''Se crea una lista para almacenar las pérdidas del crítico.'''
        for log_prob, value, ret in history:
            diff = ret - value
            '''Se calcula la diferencia entre el valor de retorno real y el valor estimado por el crítico.'''
            actor_losses.append(-log_prob * diff)  # actor loss
            '''Se agrega la pérdida del actor a la lista actor_losses, que es el producto del logaritmo negativo de la probabilidad de acción y la diferencia entre el valor de retorno real y el valor del crítico.'''
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )
            '''Se agrega la pérdida del crítico a la lista critic_losses. La pérdida del crítico se calcula utilizando la función de pérdida Huber y compara el valor estimado por el crítico con el valor de retorno real.'''

        #retropropagación
        loss_value = sum(actor_losses) + sum(critic_losses)
        '''Se calcula la pérdida total sumando las pérdidas del actor y las pérdidas del crítico.'''


        grads = tape.gradient(loss_value, model.trainable_variables)
        '''Se calculan los gradientes de la pérdida con respecto a las variables entrenables del modelo utilizando la cinta de gradiente.'''
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        '''Se aplica el gradiente calculado al actualizar los pesos del modelo utilizando el optimizador.'''

        #Borrar el historial de pérdidas y recompensas
        action_probs_history.clear()
        '''Se borra el historial de probabilidades de acción.'''
        critic_value_history.clear()
        ''' Se borra el historial de valores del crítico.'''
        rewards_history.clear()
        '''Se borra el historial de recompensas.'''

    #Detalles de registro
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        '''Se define una plantilla de cadena para imprimir el running_reward y el número de episodio.'''
        print(template.format(running_reward, episode_count)) # Se imprime el running_reward y el número de episodio utilizando la plantilla de cadena.

    if running_reward > 195:  # Condición para considerar que se ha resuelto la tarea  Si el running_reward supera el umbral de 195, se considera que el problema está resuelto y se ejecuta el siguiente bloque de código.
        print("Solved at episode {}!".format(episode_count))
        break



# Después de ejecutar los pasos del episodio, se actualiza la recompensa 
# acumulada promediada en running_reward. A continuación, se calculan los 
# valores esperados de las recompensas utilizando el factor de descuento gamma. 
# Estos valores se utilizan como etiquetas para el crítico. Luego, los 
# historiales se normalizan y se calculan las pérdidas del actor y el crítico.

# Después del cálculo de las pérdidas, se realiza la retropropagación y la 
# actualización de los parámetros del modelo utilizando el optimizador Adam. 
# Luego, se borran los historiales de probabilidades de acción, valores del 
# crítico y recompensas para el siguiente episodio.

# Se registran los detalles del episodio, como la recompensa acumulada promediada, 
# y se verifica si se ha alcanzado la condición para considerar que se ha resuelto 
# la tarea (recompensa promediada mayor a 195). Si se cumple la condición, 
# se imprime un mensaje y se rompe el bucle de entrenamiento.

# En resumen, este código implementa el algoritmo Actor-Critic para entrenar 
# un modelo que resuelva el entorno "CartPole-v0" utilizando TensorFlow y Keras.