import pandas as pd

poly = 0x4599
mask = 0x7FFF

def crc15(byte):
    crc = 0

    for bit in byte:
        crc = crc ^ (bit << 14)
        if crc & 0x4000:
            crc = ((crc << 1) ^ poly) & mask
        else:
            crc = (crc << 1) & mask
    return crc


def parse_data(data_str):
    if pd.isna(data_str):
        return []
    return [int(x, 16) for x in data_str.split()]


def build_bits(row):
    bits = []

    bits.append(0)
    id_val = int(row["id"], 16)
    for i in range(10, -1, -1):
        bits.append((id_val >> i) & 1)

    bits.append(int(row["rtr"]))
    bits.append(int(row["ide"]))
    bits.append(0)

    dlc = int(row["dlc"])
    for i in range(3, -1, -1):
        bits.append((dlc >> i) & 1)

    for byte in parse_data(row["data"]):
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    return bits

def validate_file(csv_file):
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        try:
            bits = build_bits(row)

            crc_calc = crc15(bits)
            crc_given = int(row["crc"], 16)

            timestamp = row["timestamp"]
            given_error = row["errors"]

            if crc_calc == crc_given:
                status = "success"
                error_type = "none"
            else:
                status = "failed"
                error_type = "crc error"

            print(f"{timestamp}: The CAN frame check is {status}. "
                  f"(error: {error_type}). The given error is {given_error}")

        except Exception as e:
            print(f"{row['timestamp']}: The CAN frame check is failed. "
                  f"(error: bad_frame). The given error is {row['errors']}")

validate_file("can_frames.csv")