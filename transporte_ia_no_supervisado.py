# transporte_ia.py

import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransporteMasivoIA:
    def __init__(self):
        self.grafo = nx.DiGraph()  #2
        self.estaciones_concurridas = set()
        self.estaciones_mantenimiento = set()
        self.model = self.crear_modelo()

    def agregar_estacion(self, nombre): #3
        self.grafo.add_node(nombre)

    def agregar_conexion(self, origen, destino, tiempo, concurrencia=0, transbordo=False):
        self.grafo.add_edge(origen, destino, tiempo=tiempo, concurrencia=concurrencia, transbordo=transbordo)

    def definir_estaciones_concurridas(self, estaciones):
        self.estaciones_concurridas.update(estaciones)

    def definir_estaciones_mantenimiento(self, estaciones):
        self.estaciones_mantenimiento.update(estaciones)

    def crear_modelo(self): #4
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(3,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def entrenar_modelo(self, datos, etiquetas):
        datos_np = np.array(datos)
        etiquetas_np = np.array(etiquetas)
        self.model.fit(datos_np, etiquetas_np, epochs=100, verbose=0)

    def predecir_ruta(self, tiempo, concurrencia, transbordos): #5
        entrada = np.array([[tiempo, concurrencia, transbordos]])
        return self.model.predict(entrada, verbose=0)[0][0]

    def encontrar_mejores_rutas(self, origen, destino, preferencia, max_rutas=3): #6
        try:
            rutas = list(nx.all_simple_paths(self.grafo, origen, destino))
            rutas_puntuadas = []

            for ruta in rutas:
                tiempo_total = sum(self.grafo[u][v]['tiempo'] for u, v in zip(ruta, ruta[1:]))
                concurrencia_total = sum(self.grafo[u][v]['concurrencia'] for u, v in zip(ruta, ruta[1:]))
                transbordos_total = sum(1 for u, v in zip(ruta, ruta[1:]) if self.grafo[u][v]['transbordo'])

                if preferencia == "rapidez":
                    puntaje = tiempo_total
                elif preferencia == "menos_transbordos":
                    puntaje = transbordos_total
                elif preferencia == "menos_concurrida":
                    puntaje = concurrencia_total
                elif preferencia == "evitar_mantenimiento":
                    if any(est in self.estaciones_mantenimiento for est in ruta):
                        continue
                    puntaje = self.predecir_ruta(tiempo_total, concurrencia_total, transbordos_total)
                else:
                    puntaje = self.predecir_ruta(tiempo_total, concurrencia_total, transbordos_total)

                rutas_puntuadas.append((ruta, puntaje))

            rutas_puntuadas.sort(key=lambda x: x[1])
            mejores_rutas = [ruta for ruta, _ in rutas_puntuadas[:max_rutas]]
            return mejores_rutas if mejores_rutas else ["No hay rutas disponibles"]
        except nx.NetworkXNoPath:
            return ["No hay rutas disponibles"]
        except nx.NodeNotFound:
            return ["Estación no encontrada"]

# --------------------------
# EJECUCIÓN DEL PROGRAMA
# --------------------------

transporte = TransporteMasivoIA()

# Estaciones
estaciones = ["A", "B", "C", "D", "E", "F", "G"]
for estacion in estaciones:
    transporte.agregar_estacion(estacion)

# Conexiones
conexiones = [
    ("A", "B", 5, 10, False),
    ("B", "C", 7, 20, True),
    ("C", "D", 6, 5, False),
    ("D", "E", 8, 15, True),
    ("E", "F", 4, 30, False),
    ("F", "G", 3, 25, False),
    ("A", "C", 10, 15, False),
    ("B", "D", 8, 12, True),
    ("C", "E", 9, 18, True),
    ("D", "F", 7, 22, False)
]
for conexion in conexiones:
    transporte.agregar_conexion(*conexion)

# Estaciones clave
transporte.definir_estaciones_concurridas(["C", "E"])
transporte.definir_estaciones_mantenimiento(["D"])

# Datos sintéticos de entrenamiento
datos_entrenamiento = [
    [5, 10, 0], [7, 20, 1], [6, 5, 0],
    [8, 15, 1], [4, 30, 0], [3, 25, 0]
]
etiquetas = [5, 7, 6, 8, 4, 3]  # Tiempos históricos ideales
transporte.entrenar_modelo(datos_entrenamiento, etiquetas)

# Preferencias de usuarios
usuarios = {
    "Viviana": "rapidez",
    "Diana": "menos_transbordos",
    "Andres": "menos_concurrida",
    "Mauricio": "evitar_mantenimiento"
}

# Interacción
while True:
    nombre_usuario = input("Ingrese su nombre (Viviana, Diana, Andres, Mauricio): ").strip()

    if nombre_usuario in usuarios:
        preferencia = usuarios[nombre_usuario]
        rutas = transporte.encontrar_mejores_rutas("A", "G", preferencia, max_rutas=3)

        print(f"\nRutas óptimas para {nombre_usuario} ({preferencia}):")
        for i, ruta in enumerate(rutas, 1):
            print(f"Ruta {i}: {ruta}")
        break
    else:
        print("Nombre no válido. Intente de nuevo.")