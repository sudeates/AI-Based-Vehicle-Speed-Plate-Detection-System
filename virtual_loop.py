"""
virtual_loop.py
----------------
"Sanal Radar" (Virtual Loop) mantigi: her karede kucuk, gurultulu bir hiz
olcumu yapmak yerine, aracin giris cizgisini (entry_y) ve cikis cizgisini
(exit_y) hangi FRAME'de gectigini interpolasyonla bulup, ikisi arasindaki
TEK bir makro hiz degerini hesaplar. Bu yontem tekil kare olculerindeki
titresim/hata payini buyuk olcude azaltir.
"""

from config import LOOP_BOUNDS, FAMILY_SPLIT_Y, MANUAL_DISTANCES_M


class VirtualLoop:
    """Her arac (track_id) icin, ilk gorulen y konumuna gore dogru grubu
    (A/B) secer ve o grubun giris/cikis sinirlarini kullanir.
    entry_y'yi gectigi (interpole edilmis) ani ve exit_y'yi gectigi ani
    kaydedip aradaki mesafe/zamandan tek bir hiz hesaplar."""

    def __init__(self, fps, debug=False):
        self.fps = fps
        self.debug = debug

        self.family = {}           # track_id -> "A" / "B"
        self.last_position = {}    # track_id -> (frame_idx, y)
        self.entry_frame = {}      # track_id -> interpole edilmis giris frame'i
        self.speeds = {}           # track_id -> hesaplanan hiz (km/h)
        
    def _get_bounds(self, family):
        entry_y, exit_y = LOOP_BOUNDS[family]
        real_distance_m = MANUAL_DISTANCES_M[family]
        return entry_y, exit_y, real_distance_m

    def update(self, track_id, frame_idx, y):
        """Yeni bir kare geldiginde cagrilir. Arac icin hiz zaten hesaplandiysa
        (self.speeds icinde varsa) hicbir sey yapmaz."""
        if track_id in self.speeds:
            return

        if track_id not in self.family:
            self.family[track_id] = "A" if y < FAMILY_SPLIT_Y else "B"
        entry_y, exit_y, real_distance_m = self._get_bounds(self.family[track_id])

        prev = self.last_position.get(track_id)
        self.last_position[track_id] = (frame_idx, y)
        if prev is None:
            return
        prev_frame, prev_y = prev

        # --- Giris cizgisi kontrolu ---
        if track_id not in self.entry_frame:
            if prev_y > entry_y and frame_idx < 10 and self.debug:
                print(f"!!! UYARI: ID {track_id} baslangic cizgisini kacirdi! "
                      f"Ilk y={prev_y:.1f}, Beklenen giris={entry_y}")

            if prev_y < entry_y <= y:
                fraction = (entry_y - prev_y) / (y - prev_y) if y != prev_y else 0
                self.entry_frame[track_id] = prev_frame + fraction
                if self.debug:
                    print(f"[LOOP] track={track_id} grup={self.family[track_id]} "
                          f"GIRIS tespit edildi (frame~{self.entry_frame[track_id]:.1f})")
            return

        # --- Cikis cizgisi kontrolu ---
        if prev_y < exit_y <= y:
            fraction = (exit_y - prev_y) / (y - prev_y) if y != prev_y else 0
            exit_frame = prev_frame + fraction
            elapsed_frames = exit_frame - self.entry_frame[track_id]
            if self.debug:
                print(f"[LOOP] track={track_id} grup={self.family[track_id]} "
                      f"CIKIS tespit edildi (frame~{exit_frame:.1f}, gecen_frame={elapsed_frames:.1f})")
            if elapsed_frames > 0:
                elapsed_s = elapsed_frames / self.fps
                speed_ms = real_distance_m / elapsed_s
                self.speeds[track_id] = speed_ms * 3.6

    def get(self, track_id):
        return self.speeds.get(track_id)
