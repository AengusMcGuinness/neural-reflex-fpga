"""Stable, little-endian USB PipeIn packet contract for 16 raw EMG samples."""

import struct

MAGIC = 0x4E524658  # "NRFX" when displayed as a 32-bit hexadecimal word
SAMPLES_PER_BATCH = 16
CHANNELS = 8
WORDS_PER_SAMPLE = CHANNELS // 2
PACKET_BYTES = 16 + SAMPLES_PER_BATCH * WORDS_PER_SAMPLE * 4


def pack_batch(sequence, band_timestamp_us, values):
    if not isinstance(sequence, int) or isinstance(sequence, bool) or not 0 <= sequence < 2**32:
        raise ValueError("sequence must be a uint32")
    if (not isinstance(band_timestamp_us, int) or isinstance(band_timestamp_us, bool)
            or not 0 <= band_timestamp_us < 2**64):
        raise ValueError("band timestamp must be a uint64 microsecond value")
    if not isinstance(values, list) or len(values) != SAMPLES_PER_BATCH:
        raise ValueError("a packet must contain exactly 16 samples")
    words = []
    for sample in values:
        if not isinstance(sample, list) or len(sample) != CHANNELS:
            raise ValueError("every sample must have eight channels")
        if any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**16
               for value in sample):
            raise ValueError("Kinesis raw ADC counts must be uint16")
        for channel in range(0, CHANNELS, 2):
            words.append(sample[channel] | sample[channel + 1] << 16)
    return struct.pack("<IIQ", MAGIC, sequence, band_timestamp_us) + struct.pack(
        "<64I", *words)


def unpack_batch(packet):
    if len(packet) != PACKET_BYTES:
        raise ValueError(f"packet must be {PACKET_BYTES} bytes")
    magic, sequence, band_timestamp_us = struct.unpack_from("<IIQ", packet)
    if magic != MAGIC:
        raise ValueError("packet magic mismatch")
    words = struct.unpack_from("<64I", packet, 16)
    values = []
    for sample in range(SAMPLES_PER_BATCH):
        channels = []
        for word in words[sample * WORDS_PER_SAMPLE:(sample + 1) * WORDS_PER_SAMPLE]:
            channels.extend((word & 0xFFFF, word >> 16))
        values.append(channels)
    return sequence, band_timestamp_us, values
