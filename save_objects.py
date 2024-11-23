import pickle

# Pastikan tokenizer dan scaler sudah didefinisikan sebelumnya atau diimpor
from training_script import tokenizer, scaler  # Ubah 'training_script' sesuai nama file pelatihan Anda

# Simpan tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Simpan scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Tokenizer dan Scaler berhasil disimpan!")
