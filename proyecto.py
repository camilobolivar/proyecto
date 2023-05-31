#Setup
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Factor de descuento para recompensas pasadas
max_steps_per_episode = 10000
env = gym.make("CartPole-v0")  # Crea el entorno CartPole-v0
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # El número más pequeño tal que 1.0 + eps != 1.0


# En esta sección se importan las bibliotecas necesarias
# y se establecen los parámetros de configuración, como 
# la semilla para reproducibilidad, el factor de descuento 
# gamma y el número máximo de pasos por episodio. También 
# se crea el entorno CartPole-v0 y se establece una semilla para el entorno.



#Implement Actor Critic network

num_inputs = 4
num_actions = 2
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])


#En esta sección se implementa la red Actor-Critic. 
# La red consta de tres capas: una capa de entrada con 
# forma (num_inputs,), una capa oculta completamente 
# conectada con activación ReLU y una capa de salida para 
# las acciones con activación softmax, y una capa de salida 
# para el crítico que estima las recompensas futuras. 
# El modelo se crea utilizando la API funcional de Keras.


# Train

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:  # Ejecuta hasta que se resuelva
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Agregar esta línea mostraría los intentos
            # del agente en una ventana emergente.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predecir probabilidades de acción y valores estimados de recompensa futura
            # a partir del estado del entorno
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Muestrear acción de la distribución de probabilidad de acción
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Aplicar la acción muestreada en nuestro entorno
            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

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


        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condición para considerar que se ha resuelto la tarea
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