import tkinter as tk
from tkinter import simpledialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functions import plot_double_pendulum, double_pendulum_equations, create_double_pendulum_simulation, process_gif

def run_gui():
    root = tk.Tk()
    root.title("Double Pendulum Analysis")

    def start_simulation():
        num_pendulums = simpledialog.askinteger("Input", "Kaç adet double pendulum grafiği analizi yapmak istiyorsunuz?")
        
        if num_pendulums is None:
            return
        
        # Simülasyon parametreleri
        L1, L2 = 1.0, 1.0
        m1, m2 = 1.0, 1.0
        y0 = [np.pi / 2, 0, np.pi / 2, 0]  # Başlangıç koşulları
        t_span = [0, 10]
        t_eval = np.linspace(0, 10, 300)

        for i in range(num_pendulums):
            sol = create_double_pendulum_simulation(L1, L2, m1, m2, y0, t_span, t_eval)
            theta1, z1, theta2, z2 = sol
            x1, y1, x2, y2 = plot_double_pendulum(L1, L2, theta1, theta2)

            fig, ax = plt.subplots()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            line, = ax.plot([], [], 'o-', lw=2)

            def update(frame):
                line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
                return line,

            ani = FuncAnimation(fig, update, frames=len(t_eval), blit=True)
            ani.save(f'assets/double_pendulum_{i}.gif', writer='imagemagick')
            plt.close(fig)

        coords = []
        for i in range(num_pendulums):
            x, y = process_gif(f'assets/double_pendulum_{i}.gif')
            coords.append((x, y))

        for i, (x_coords, y_coords) in enumerate(coords):
            plt.figure()
            plt.plot(x_coords, y_coords, 'bo-')
            plt.xlabel('X Koordinatı')
            plt.ylabel('Y Koordinatı')
            plt.title(f'Double Pendulum {i+1} Uç Noktası Hareketi')
            plt.grid()
            plt.savefig(f'results/double_pendulum_{i}_analysis.png')
            plt.show()

        messagebox.showinfo("Bilgi", "Analiz tamamlandı ve sonuçlar kaydedildi.")

    start_button = tk.Button(root, text="Simülasyonu Başlat", command=start_simulation)
    start_button.pack(pady=20)

    root.mainloop()
