import numpy as np
from modulo_components.lifecycle_component import LifecycleComponent
import state_representation as sr

class JtcCommandGenerator(LifecycleComponent):
    def __init__(self, node_name: str, *args, **kwargs):
        super().__init__(node_name, *args, **kwargs)

        # --- PARAMETER ---
        # Geschwindigkeit in m/s (Standard 0.1), in der AICA UI anpassbar
        self.add_parameter(sr.Parameter("v_max", 0.1))
        # Der Name des Frames, den der "Signal to TF"-Block erzeugt
        self.add_parameter(sr.Parameter("target_tf_name", "mein_ziel_frame"))

        # --- INPUTS ---
        self.ist_pose = sr.CartesianPose("ist_pose", "world")
        self.add_input("_ist_pose", "ist_pose", sr.CartesianPose)

        self.target_pose = sr.CartesianPose("target_pose", "world")
        self.add_input("_target_pose", "target_pose", sr.CartesianPose)

        # --- OUTPUTS ---
        self.jtc_command = ""
        self.add_output("_jtc_command", "jtc_command", str)

        # --- INTERNER STATUS ---
        self._last_target_pos = None

    def on_step_callback(self):
        # 1. Schutzabfrage: Warten bis echte Roboterdaten da sind
        if self.ist_pose.is_empty() or self.target_pose.is_empty():
            return

        current_target_pos = self.target_pose.get_position()

        # 2. Flankenauswertung (Wurde eine NEUE Pose geschickt?)
        if self._last_target_pos is not None:
            # Wenn der Unterschied zum letzten Ziel < 1mm ist, brich ab
            dist_to_last_target = np.linalg.norm(current_target_pos - self._last_target_pos)
            if dist_to_last_target < 0.001:
                return 

        # 3. Mathematik: Euklidische Distanz vom aktuellen TCP zum neuen Ziel
        robot_pos = self.ist_pose.get_position()
        distance = np.linalg.norm(current_target_pos - robot_pos)

        # 4. Dauer berechnen (Zeit = Strecke / Geschwindigkeit)
        v_max = self.get_parameter("v_max").get_value()
        
        # Fallback falls v_max versehentlich auf 0 gesetzt wird
        if v_max <= 0.0:
            v_max = 0.1 
            
        duration = distance / v_max
        
        # Optionales Sicherheits-Limit: Niemals schneller als 0.5 Sekunden ankommen
        duration = max(duration, 0.5) 

        # 5. String-Befehl zusammenbauen
        frame_name = self.get_parameter("target_tf_name").get_value()
        
        # Erzeugt exakt: {frames: [Name], durations: [Zeit]}
        self.jtc_command = f"{{frames: [{frame_name}], durations: [{duration:.2f}]}}"
        
        self.get_logger().info(f"Neues Ziel erkannt! Distanz: {distance:.3f}m. Sende Befehl: {self.jtc_command}")

        # 6. Status aktualisieren (damit der Befehl nicht nochmal gesendet wird)
        self._last_target_pos = current_target_pos.copy()