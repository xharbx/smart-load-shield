"""
IEEE 26-Bus Power System Model in pandapower
=============================================
Data extracted from standard IEEE 26-bus test case tables.
Sbase = 100 MVA, Vbase = 230 kV (assumed transmission voltage)

Author: Salah (built for Prof Ahmad Harb research)
"""

import pandapower as pp
import numpy as np


def build_ieee26():
    """Build and return the IEEE 26-bus pandapower network."""
    net = pp.create_empty_network(name="IEEE 26-Bus System", sn_mva=100.0)

    # =========================================================================
    # System base values
    # =========================================================================
    S_BASE = 100.0   # MVA
    V_BASE = 230.0   # kV  (assumed nominal transmission voltage)
    Z_BASE = V_BASE**2 / S_BASE  # = 529 ohm

    # =========================================================================
    # BUS DATA  (26 buses, numbered 1-26)
    # =========================================================================
    # Generator buses with specified voltages
    gen_v = {1: 1.025, 2: 1.020, 3: 1.025, 4: 1.050, 5: 1.045, 26: 1.015}

    for i in range(1, 27):
        bus_type = "b"  # default PQ bus
        vn_kv = V_BASE
        if i == 1:
            bus_type = "b"  # will be set as ext_grid (slack)
        pp.create_bus(net, vn_kv=vn_kv, name=f"Bus {i}", index=i)

    # =========================================================================
    # LOAD DATA  (from image_1.png)
    # =========================================================================
    load_data = [
        # (bus, P_MW, Q_Mvar)
        (1,  51.0, 41.0),
        (2,  22.0, 15.0),
        (3,  64.0, 50.0),
        (4,  25.0, 10.0),
        (5,  50.0, 30.0),
        (6,  76.0, 29.0),
        (7,   0.0,  0.0),
        (8,   0.0,  0.0),
        (9,  89.0, 50.0),
        (10,  0.0,  0.0),
        (11, 25.0, 15.0),
        (12, 89.0, 48.0),
        (13, 31.0, 15.0),
        (14, 24.0, 12.0),
        (15, 70.0, 31.0),
        (16, 55.0, 27.0),
        (17, 78.0, 38.0),
        (18,153.0, 67.0),
        (19, 75.0, 15.0),
        (20, 48.0, 27.0),
        (21, 46.0, 23.0),
        (22, 45.0, 22.0),
        (23, 25.0, 12.0),
        (24, 54.0, 27.0),
        (25, 28.0, 13.0),
        (26, 40.0, 20.0),
    ]

    for bus, p_mw, q_mvar in load_data:
        if p_mw > 0 or q_mvar > 0:
            pp.create_load(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar,
                           name=f"Load Bus {bus}")

    # =========================================================================
    # GENERATION DATA  (from image_2.png)
    # =========================================================================
    # Bus 1: Slack bus, V = 1.025 pu
    pp.create_ext_grid(net, bus=1, vm_pu=1.025, va_degree=0.0,
                       name="Slack Bus 1")

    # Other generators (PV buses)
    gen_data = [
        # (bus, P_MW, V_pu, Qmin, Qmax)
        (2,   79.0, 1.020,  40.0, 250.0),
        (3,   20.0, 1.025,  40.0, 150.0),
        (4,  100.0, 1.050,  40.0,  80.0),
        (5,  300.0, 1.045,  40.0, 160.0),
        (26,  60.0, 1.015,  15.0,  50.0),
    ]

    for bus, p_mw, vm_pu, qmin, qmax in gen_data:
        pp.create_gen(net, bus=bus, p_mw=p_mw, vm_pu=vm_pu,
                      min_q_mvar=-qmin, max_q_mvar=qmax,
                      name=f"Gen Bus {bus}")

    # =========================================================================
    # SHUNT CAPACITORS  (from image_3.png)
    # =========================================================================
    shunt_data = [
        # (bus, Q_Mvar)  — positive Q means capacitive injection
        (1,   4.0),
        (4,   2.0),
        (5,   5.0),
        (6,   2.0),
        (11,  1.5),
        (12,  2.0),
        (15,  0.5),
        (19,  5.0),
    ]

    for bus, q_mvar in shunt_data:
        # In pandapower, shunt q_mvar is negative for capacitive (generating reactive)
        pp.create_shunt(net, bus=bus, q_mvar=-q_mvar, p_mw=0.0,
                        name=f"Shunt Cap Bus {bus}")

    # =========================================================================
    # TRANSFORMER TAP DATA  (from image_3.png)
    # =========================================================================
    transformer_taps = {
        (2, 3):   0.960,
        (2, 13):  0.960,
        (3, 13):  1.017,
        (4, 8):   1.050,
        (4, 12):  1.050,
        (6, 19):  0.950,
        (7, 9):   0.950,
    }

    # =========================================================================
    # LINE AND TRANSFORMER DATA  (from image_4_large.png)
    # =========================================================================
    # Format: (from_bus, to_bus, R_pu, X_pu, half_B_pu)
    # Branches with half_B = 0 and listed in transformer_taps are transformers.
    branch_data = [
        # Left column of the table
        (1,   2,  0.0005, 0.0048, 0.0300),
        (1,  18,  0.0013, 0.0110, 0.0600),
        (2,   3,  0.0014, 0.0513, 0.0500),   # Transformer
        (2,   7,  0.0103, 0.0586, 0.0180),
        (2,   8,  0.0074, 0.0321, 0.0390),
        (2,  13,  0.0035, 0.0967, 0.0250),   # Transformer
        (2,  26,  0.0323, 0.1967, 0.0000),
        (3,  13,  0.0007, 0.0054, 0.0005),   # Transformer
        (4,   8,  0.0008, 0.0240, 0.0001),   # Transformer
        (4,  12,  0.0016, 0.0207, 0.0150),   # Transformer
        (5,   6,  0.0069, 0.0300, 0.0990),
        (6,  11,  0.0053, 0.0306, 0.0010),
        (6,  18,  0.0097, 0.0570, 0.0001),
        (6,  19,  0.0037, 0.0222, 0.0012),   # Transformer
        (7,   9,  0.0035, 0.0660, 0.0450),   # Transformer
        (7,   8,  0.0012, 0.0069, 0.0491),
        (8,  12,  0.0020, 0.0180, 0.0000),
        (9,  10,  0.0010, 0.0493, 0.0010),
        (10, 19,  0.0547, 0.2360, 0.0000),
        (10, 20,  0.0066, 0.0160, 0.0010),

        # Right column of the table
        (10, 22,  0.0069, 0.0298, 0.0050),
        (11, 25,  0.0960, 0.2700, 0.0100),
        (11, 26,  0.0165, 0.0970, 0.0040),
        (12, 14,  0.0327, 0.0802, 0.0000),
        (12, 15,  0.0180, 0.0598, 0.0020),
        (13, 14,  0.0046, 0.0271, 0.0001),
        (13, 15,  0.0116, 0.0610, 0.0000),
        (13, 16,  0.0179, 0.0888, 0.0000),
        (14, 15,  0.0069, 0.0382, 0.0000),
        (15, 16,  0.0209, 0.0512, 0.0000),
        (16, 17,  0.0990, 0.0000, 0.0000),   # See note
        (17, 18,  0.0032, 0.0600, 0.0380),
        (17, 21,  0.0290, 0.4450, 0.0000),   # Possibly 0.0290/0.4450
        (19, 23,  0.0300, 0.1310, 0.0000),
        (19, 24,  0.0300, 0.1250, 0.0020),
        (19, 25,  0.1190, 0.2249, 0.0040),
        (20, 21,  0.0657, 0.1570, 0.0000),
        (20, 22,  0.0150, 0.0366, 0.0000),
        (21, 24,  0.0476, 0.1510, 0.0000),
        (22, 24,  0.0310, 0.0880, 0.0000),
        (23, 25,  0.0987, 0.1168, 0.0000),
    ]

    # NOTE on row 16-17: The image shows R=0.0990, X=0.0000 which is unusual.
    # Re-reading: it looks like (16,17): R=0.0990, X=0.0000 -- but X=0 is
    # problematic. Let me re-interpret from the image more carefully.
    # Looking again at the right column rows:
    # 16 17 0.0990 0.0000 0.0000 seems wrong.
    # Actually re-reading: the row for 16-17 shows 0.0990 0.0585 0.0000
    # Let me also re-check row 17-18 and 17-21.
    # After careful re-examination:
    # 16-17: R=0.0990, X=0.0585 (not 0.0000), B/2=0.000  -- CORRECTED BELOW
    # 17-18: R=0.0032, X=0.0600, B/2=0.038
    # 17-20: R=0.0239, X=0.0585, B/2=0.000  -- this might be another row
    # 17-21: R=0.0290, X=0.4450, B/2=0.000  -- possibly R=0.0290

    # Also check for the duplicate (7,9) entry. The table shows two rows:
    # Row 1: 7-9, 0.0035, 0.0660, 0.0450  (this is the transformer)
    # Row 2: 7-9, 0.0009, 0.0429, 0.0250  (possibly a second parallel branch)
    # Looking at the image again more carefully, the second row might be 8-12
    # or another branch. But the image clearly shows 7-9 twice. Let me keep
    # only the transformer entry (first one) and treat the second as a
    # parallel line if it exists.

    # Let me rebuild the branch list more carefully, fixing the issues:

    branch_data_clean = []

    # I'll process each branch. For the (16,17) fix:
    for i, (fb, tb, r, x, b_half) in enumerate(branch_data):
        if fb == 16 and tb == 17 and x == 0.0:
            # Correct: from re-reading image, X should be ~0.0585
            branch_data_clean.append((16, 17, 0.0239, 0.0585, 0.0000))
        else:
            branch_data_clean.append((fb, tb, r, x, b_half))

    # Also re-read more carefully. Let me just re-do the right column properly:
    # Actually, let me re-examine the image one more time. The right side rows are:
    #
    # 10 22  0.0069  0.0298  0.005
    # 11 25  0.0960  0.2700  0.010
    # 11 26  0.0165  0.0970  0.004
    # 12 14  0.0327  0.0802  0.000
    # 12 15  0.0180  0.0598  0.002
    # 13 14  0.0046  0.0271  0.001
    # 13 15  0.0116  0.0610  0.000
    # 13 16  0.0179  0.0888  0.000  -- possibly 0.0179 0.0888
    # 14 15  0.0069  0.0382  0.000
    # 15 16  0.0209  0.0512  0.000
    # 16 17  0.0990  0.0000  0.000  -- X=0 is suspicious
    # 17 18  0.0032  0.0600  0.038
    # 17 21  0.0290  0.4450  0.000  -- very high X, might be 0.0290/0.0445
    # 19 23  0.0300  0.1310  0.000
    # 19 24  0.0300  0.1250  0.002
    # 19 25  0.1190  0.2249  0.004
    # 20 21  0.0657  0.1570  0.000
    # 20 22  0.0150  0.0366  0.000
    # 21 24  0.0476  0.1510  0.000
    # 21 23  0.0290  0.0990  0.000
    # 22 24  0.0310  0.0880  0.000
    # 23 25  0.0987  0.1168  0.000

    # I'll use the final cleaned list directly below.

    # =========================================================================
    # CREATE LINES AND TRANSFORMERS
    # =========================================================================
    # Final branch data after careful reading:
    branches = [
        # (from, to,   R_pu,    X_pu,    B/2_pu)
        (1,   2,  0.0005, 0.0048, 0.0300),
        (1,  18,  0.0013, 0.0110, 0.0600),
        (2,   3,  0.0014, 0.0513, 0.0500),   # Transformer tap=0.960
        (2,   7,  0.0103, 0.0586, 0.0180),
        (2,   8,  0.0074, 0.0321, 0.0390),
        (2,  13,  0.0035, 0.0967, 0.0250),   # Transformer tap=0.960
        (2,  26,  0.0323, 0.1967, 0.0000),
        (3,  13,  0.0007, 0.0054, 0.0005),   # Transformer tap=1.017
        (4,   8,  0.0008, 0.0240, 0.0001),   # Transformer tap=1.050
        (4,  12,  0.0016, 0.0207, 0.0150),   # Transformer tap=1.050
        (5,   6,  0.0069, 0.0300, 0.0990),
        (6,  11,  0.0097, 0.0570, 0.0001),   # Fixed 2026-04-25 (was 0.0053/0.0306/0.0010, copied from 6-7)
        (6,  18,  0.0037, 0.0222, 0.0012),   # Fixed 2026-04-25 (was 0.0097/0.0570/0.0001, copied from 6-11)
        (6,  19,  0.0035, 0.0660, 0.0450),   # Transformer tap=0.950, fixed 2026-04-25 (was 0.0037/0.0222/0.0012)
        (7,   9,  0.0009, 0.0429, 0.0250),   # Transformer tap=0.950, fixed 2026-04-25 (was 0.0035/0.0660/0.0450, copied from 6-19)
        (7,   8,  0.0012, 0.0069, 0.0001),   # Fixed 2026-04-25 B/2 (was 0.0491)
        (8,  12,  0.0020, 0.0180, 0.0200),   # Fixed 2026-04-25 B/2 (was 0)
        (9,  10,  0.0010, 0.0493, 0.0010),
        (10, 19,  0.0547, 0.2360, 0.0000),
        (10, 20,  0.0066, 0.0160, 0.0010),
        (10, 22,  0.0069, 0.0298, 0.0050),
        (11, 25,  0.0960, 0.2700, 0.0100),
        (11, 26,  0.0165, 0.0970, 0.0040),
        (12, 14,  0.0327, 0.0802, 0.0000),
        (12, 15,  0.0180, 0.0598, 0.0020),
        (13, 14,  0.0046, 0.0271, 0.0001),
        (13, 15,  0.0116, 0.0610, 0.0000),
        (13, 16,  0.0179, 0.0888, 0.0000),
        (14, 15,  0.0069, 0.0382, 0.0000),
        (15, 16,  0.0209, 0.0512, 0.0000),
        (16, 17,  0.0990, 0.0600, 0.0000),   # Fixed 2026-04-25 (was 0.0239/0.0585, that's actually 16-20's data)
        (17, 18,  0.0032, 0.0600, 0.0380),
        (17, 21,  0.2290, 0.4450, 0.0000),   # Fixed 2026-04-25 (was 0.0290/0.0445, off by 10x)
        (19, 23,  0.0300, 0.1310, 0.0000),
        (19, 24,  0.0300, 0.1250, 0.0020),
        (19, 25,  0.1190, 0.2249, 0.0040),
        (20, 21,  0.0657, 0.1570, 0.0000),
        (20, 22,  0.0150, 0.0366, 0.0000),
        (21, 24,  0.0476, 0.1510, 0.0000),
        (22, 24,  0.0310, 0.0880, 0.0000),
        (23, 25,  0.0987, 0.1168, 0.0000),
    ]

    # Track which branches are transformers
    trafo_keys = set(transformer_taps.keys())

    # Keep track of the first 7-9 branch (transformer) vs second (line)
    seen_7_9 = False

    for fb, tb, r_pu, x_pu, b_half_pu in branches:
        key = (fb, tb)
        is_trafo = key in trafo_keys

        # Special handling: first (7,9) is transformer, second is parallel line
        if key == (7, 9):
            if not seen_7_9:
                is_trafo = True
                seen_7_9 = True
            else:
                is_trafo = False

        if is_trafo:
            tap = transformer_taps[key]
            # Create transformer from parameters
            # pandapower transformer model:
            # vn_hv_kv, vn_lv_kv = both V_BASE (same voltage level tap-changing)
            # vk_percent = sqrt(r^2 + x^2) * 100  (short circuit voltage %)
            # vkr_percent = r * 100
            # For in-line transformers (same voltage), use same vn on both sides
            # i0_percent and pfe_kw set to 0 (ideal core)

            vk_percent = np.sqrt(r_pu**2 + x_pu**2) * 100.0
            vkr_percent = r_pu * 100.0

            # tap_pos: pandapower uses tap ratio. tap_side = "hv"
            # tap_neutral = 0, tap_step_percent calculated from tap setting
            # For tap=0.96: deviation = (0.96 - 1.0) * 100 = -4%
            # We'll set tap_step_percent = abs(tap-1)*100, tap_pos = +/-1

            sn_mva = S_BASE  # rating = system base

            # Charging susceptance for transformers (B/2)
            # Usually small/zero for transformers; we'll ignore it for trafos

            pp.create_transformer_from_parameters(
                net,
                hv_bus=fb, lv_bus=tb,
                sn_mva=sn_mva,
                vn_hv_kv=V_BASE, vn_lv_kv=V_BASE,
                vkr_percent=vkr_percent,
                vk_percent=vk_percent,
                pfe_kw=0.0,
                i0_percent=0.0,
                tap_side="hv",
                tap_neutral=0,
                tap_pos=1 if tap != 1.0 else 0,
                tap_step_percent=abs(tap - 1.0) * 100.0 if tap != 1.0 else 1.0,
                tap_min=-10, tap_max=10,
                name=f"Trafo {fb}-{tb} (tap={tap})"
            )

            # Correct the tap_pos to get the desired ratio
            # pandapower: V_ratio = 1 + tap_pos * tap_step_percent / 100
            # We want V_ratio = tap
            # So: tap_pos * tap_step_percent/100 = tap - 1
            # With tap_step_percent = |tap-1|*100:
            #   tap_pos * |tap-1| = tap - 1
            #   tap_pos = (tap-1) / |tap-1| = sign(tap-1) = +1 or -1
            # This is already handled above since:
            #   tap > 1 => tap_pos=1, step=|tap-1|*100 => ratio = 1+(tap-1) = tap
            #   tap < 1 => tap_pos=1, step=|tap-1|*100 => ratio = 1+|tap-1| = 1+(1-tap) = 2-tap (WRONG)
            # Need to fix: for tap < 1, tap_pos should be -1
            idx = len(net.trafo) - 1
            if tap < 1.0:
                net.trafo.at[idx, "tap_pos"] = -1
            elif tap > 1.0:
                net.trafo.at[idx, "tap_pos"] = 1
            else:
                net.trafo.at[idx, "tap_pos"] = 0

        else:
            # Regular transmission line
            # Convert per-unit to physical values with length_km = 1
            length_km = 1.0
            r_ohm = r_pu * Z_BASE
            x_ohm = x_pu * Z_BASE

            # B/2 in pu => total B = 2 * (B/2) in pu
            # B_pu = B_total (susceptance in pu) = 2 * b_half_pu
            # B_siemens = B_pu / Z_BASE = B_pu * Y_BASE = B_pu * S_BASE / V_BASE^2
            # C = B / (2*pi*f)
            # c_nf_per_km = C * 1e9 / length_km
            # But pandapower uses total B/2 per km:
            # c_nf_per_km corresponds to the line capacitance
            # B_line = 2*pi*f*C_total
            # With length=1 km, c_nf_per_km = C_total_nF

            freq = 60.0  # Hz
            omega = 2.0 * np.pi * freq

            # B/2 in pu per line => B_half_siemens = b_half_pu / Z_BASE
            b_half_siemens = b_half_pu / Z_BASE
            # C_half = B_half / omega
            c_half_f = b_half_siemens / omega
            # pandapower c_nf_per_km is the full line capacitance per km
            # which gives B/2 at each end (pi model)
            # c_nf_per_km such that: B_total = omega * c_nf * 1e-9 * length
            # B/2 = omega * c_nf * 1e-9 * length / 2
            # So: c_nf = 2 * B_half / (omega * 1e-9 * length)
            # Wait, let me think again.
            # pandapower: B_total_siemens = omega * C_total = omega * c_nf_per_km * 1e-9 * length_km
            # We have B/2 in pu, so B_total_pu = 2 * b_half_pu
            # B_total_siemens = B_total_pu * (S_BASE / V_BASE^2) = B_total_pu / Z_BASE
            # omega * c_nf_per_km * 1e-9 * length_km = 2 * b_half_pu / Z_BASE
            # c_nf_per_km = 2 * b_half_pu / (Z_BASE * omega * 1e-9 * length_km)

            c_nf_per_km = 2.0 * b_half_pu / (Z_BASE * omega * 1e-9 * length_km)

            # Max current (dummy, set high)
            max_i_ka = 10.0

            pp.create_line_from_parameters(
                net,
                from_bus=fb, to_bus=tb,
                length_km=length_km,
                r_ohm_per_km=r_ohm,
                x_ohm_per_km=x_ohm,
                c_nf_per_km=c_nf_per_km,
                max_i_ka=max_i_ka,
                name=f"Line {fb}-{tb}"
            )

    # ---- Fix: add missing lines that were in the data table but got lost ----
    V_BASE = 230.0
    Z_BASE = V_BASE**2 / 100.0
    missing_lines = [
        (6, 7,  0.0053, 0.0306, 0.0010),
        (6, 21, 0.0050, 0.0900, 0.0226),
        (10,12, 0.0024, 0.0132, 0.0100),
        (16,20, 0.0239, 0.0585, 0.0000),
        (22,23, 0.0290, 0.0990, 0.0000),
    ]
    import math
    for fb, tb, r_pu, x_pu, b_half_pu in missing_lines:
        # Check if already exists
        exists = False
        for i in net.line.index:
            if (net.line.at[i,'from_bus']==fb and net.line.at[i,'to_bus']==tb) or \
               (net.line.at[i,'from_bus']==tb and net.line.at[i,'to_bus']==fb):
                exists = True; break
        if not exists:
            r_ohm = r_pu * Z_BASE
            x_ohm = x_pu * Z_BASE
            b_total = 2 * b_half_pu
            c_nf = b_total / Z_BASE / (2 * math.pi * 60) * 1e9 if b_total > 0 else 0
            pp.create_line_from_parameters(net, from_bus=fb, to_bus=tb, length_km=1.0,
                r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm, c_nf_per_km=c_nf,
                max_i_ka=10.0, name=f"Line {fb}-{tb}")

    return net


def add_bus27_pv(net, pv_mw=100.0, vm_pu=1.0):
    """
    Bus 27 PV embedded in the network (per Dr. Ahmad, 2026-04-18 meeting):
    Connect Bus 27 to Bus 20 AND Bus 21 using line 20-21 impedance values
    (r = 0.0657, x = 0.1570, bh = 0). Variable PV output: 10 - 200 MW.
    """
    S_BASE = 100.0
    V_BASE = 230.0
    Z_BASE = V_BASE**2 / S_BASE

    # Line 20-21 impedance (copied per spec)
    R_PU, X_PU, BH_PU = 0.0657, 0.1570, 0.0

    pp.create_bus(net, vn_kv=V_BASE, name="Bus 27 (PV)", index=27)
    pp.create_gen(net, bus=27, p_mw=pv_mw, vm_pu=vm_pu,
                  min_q_mvar=-100, max_q_mvar=100, name="PV Gen Bus 27")

    for to_b in (20, 21):
        pp.create_line_from_parameters(
            net,
            from_bus=27, to_bus=to_b,
            length_km=1.0,
            r_ohm_per_km=R_PU * Z_BASE,
            x_ohm_per_km=X_PU * Z_BASE,
            c_nf_per_km=0.0,
            max_i_ka=10.0,
            name=f"Line 27-{to_b} (PV tie)",
        )

    return net


def disconnect_line(net, from_bus, to_bus):
    """
    Part (c): Disconnect a transmission line between from_bus and to_bus.
    Sets the line out of service.
    """
    # Find the line
    mask = ((net.line.from_bus == from_bus) & (net.line.to_bus == to_bus)) | \
           ((net.line.from_bus == to_bus) & (net.line.to_bus == from_bus))
    line_idx = net.line.index[mask]

    if len(line_idx) == 0:
        # Check transformers too
        mask_t = ((net.trafo.hv_bus == from_bus) & (net.trafo.lv_bus == to_bus)) | \
                 ((net.trafo.hv_bus == to_bus) & (net.trafo.lv_bus == from_bus))
        trafo_idx = net.trafo.index[mask_t]
        if len(trafo_idx) > 0:
            net.trafo.loc[trafo_idx, "in_service"] = False
            print(f"Disconnected transformer {from_bus}-{to_bus}")
        else:
            print(f"No line or transformer found between Bus {from_bus} and Bus {to_bus}")
    else:
        net.line.loc[line_idx, "in_service"] = False
        print(f"Disconnected line {from_bus}-{to_bus}")

    return net


if __name__ == "__main__":
    print("=" * 70)
    print("IEEE 26-Bus Power System - pandapower Model")
    print("=" * 70)

    # Build the network
    net = build_ieee26()

    print(f"\nNetwork summary:")
    print(f"  Buses:          {len(net.bus)}")
    print(f"  Lines:          {len(net.line)}")
    print(f"  Transformers:   {len(net.trafo)}")
    print(f"  Generators:     {len(net.gen)} + 1 ext_grid (slack)")
    print(f"  Loads:          {len(net.load)}")
    print(f"  Shunts:         {len(net.shunt)}")

    # Run power flow
    print("\n--- Running Newton-Raphson Power Flow ---")
    try:
        pp.runpp(net, algorithm="nr", max_iteration=100, tolerance_mva=1e-6)
        print("Power flow CONVERGED!\n")
    except Exception as e:
        print(f"Power flow FAILED: {e}")
        print("Trying with higher tolerance...")
        try:
            pp.runpp(net, algorithm="nr", max_iteration=200, tolerance_mva=1e-4)
            print("Power flow CONVERGED (with relaxed tolerance)!\n")
        except Exception as e2:
            print(f"Still failed: {e2}")
            print("Trying with 'bfsw' algorithm...")
            try:
                pp.runpp(net, max_iteration=200, tolerance_mva=1e-4)
                print("Power flow CONVERGED!\n")
            except Exception as e3:
                print(f"All attempts failed: {e3}")
                import sys
                sys.exit(1)

    # Print bus voltages and angles
    print("--- Bus Voltages and Angles ---")
    print(f"{'Bus':>5} {'Vm (pu)':>10} {'Va (deg)':>10}")
    print("-" * 30)
    for bus_idx in sorted(net.bus.index):
        vm = net.res_bus.at[bus_idx, "vm_pu"]
        va = net.res_bus.at[bus_idx, "va_degree"]
        print(f"{bus_idx:>5} {vm:>10.4f} {va:>10.4f}")

    # Print generation summary
    print("\n--- Generation Summary ---")
    # External grid (slack)
    slack_p = net.res_ext_grid["p_mw"].sum()
    slack_q = net.res_ext_grid["q_mvar"].sum()
    print(f"  Slack (Bus 1):  P = {slack_p:8.2f} MW,  Q = {slack_q:8.2f} Mvar")

    total_gen_p = slack_p
    total_gen_q = slack_q
    for idx in net.gen.index:
        bus = net.gen.at[idx, "bus"]
        p = net.res_gen.at[idx, "p_mw"]
        q = net.res_gen.at[idx, "q_mvar"]
        print(f"  Gen Bus {bus:>2}:     P = {p:8.2f} MW,  Q = {q:8.2f} Mvar")
        total_gen_p += p
        total_gen_q += q

    print(f"\n  Total Generation: P = {total_gen_p:.2f} MW, Q = {total_gen_q:.2f} Mvar")

    # Load summary
    total_load_p = net.res_load["p_mw"].sum()
    total_load_q = net.res_load["q_mvar"].sum()
    print(f"  Total Load:       P = {total_load_p:.2f} MW, Q = {total_load_q:.2f} Mvar")

    # Shunt summary
    total_shunt_q = net.res_shunt["q_mvar"].sum()
    print(f"  Total Shunt Q:    Q = {total_shunt_q:.2f} Mvar")

    # Losses
    total_line_loss_p = net.res_line["pl_mw"].sum()
    total_line_loss_q = net.res_line["ql_mvar"].sum()
    total_trafo_loss_p = net.res_trafo["pl_mw"].sum()
    total_trafo_loss_q = net.res_trafo["ql_mvar"].sum()
    print(f"\n  Line losses:      P = {total_line_loss_p:.2f} MW, Q = {total_line_loss_q:.2f} Mvar")
    print(f"  Trafo losses:     P = {total_trafo_loss_p:.2f} MW, Q = {total_trafo_loss_q:.2f} Mvar")

    # Power balance check
    print("\n--- Power Balance Check ---")
    gen_total = total_gen_p
    load_total = total_load_p
    loss_total = total_line_loss_p + total_trafo_loss_p
    print(f"  Generation:  {gen_total:.2f} MW")
    print(f"  Load:        {load_total:.2f} MW")
    print(f"  Losses:      {loss_total:.2f} MW")
    print(f"  Mismatch:    {gen_total - load_total - loss_total:.4f} MW")
    if abs(gen_total - load_total - loss_total) < 1.0:
        print("  => Power balance OK!")
    else:
        print("  => WARNING: Power balance mismatch detected!")

    print("\n" + "=" * 70)
    print("Part (b) and (c) functions available:")
    print("  add_bus27_pv(net)       - Add Bus 27 PV (default 100 MW, range 10-200), tied to Bus 20 + Bus 21")
    print("  disconnect_line(net, from_bus, to_bus) - Disconnect a line")
    print("=" * 70)
