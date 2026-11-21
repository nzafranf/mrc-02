# Error sample: qwen0.5b_id1_0shot_n100 (qwen0.5b_id1_0shot_n100)

40 errors stratified by outcome (round robin) + 10 correct answers with the highest na_prob. Fill human_label with one of: gold_issue, multi_hop_miss, right_type_wrong_entity, wrong_type, none, other.

| # | id | outcome | question | gold | pred | q_type | human_label | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | qa_0349eeca8eb4cae07506 | false_no_answer | kapankah papua nugini melepaskan diri dari indonesia? | 1 Juli 1971 | (no answer) | time |  |  |
| 2 | qa_2235e9532c952b3717f9 | wrong_sentence | Komponen apakah yang tidak pernah gagal membunuh bakteri gram positif dan bakteri gram negatif? | Peptida antimikrobial | membunuh bakteri gram positif dan bakteri gram negatif, termasuk strain yang | entity_or_def |  |  |
| 3 | qa_0a3867b0e2c4601ee016 | boundary_too_long | Berapa luas Papua? | 808.105km | Papua memiliki luas 808.105km persegi | number |  |  |
| 4 | qa_074634b2eae86659178f | partial_overlap | Berapa luas Pulau Serangan? | panjang maksium 2,9 km dan lebar 1 km | Luas Pulau Serangan adalah 2,9 km persegi | number |  |  |
| 5 | qa_0d82790c6d3244970680 | boundary_too_short | siapakah orang tua Mahasthavira Vajragiri? | Ayahnya bernama The Tjai Gwan dan ibunya Liem Kiat Nio | The Tjai Gwan dan ibunya Liem Kiat Nio | person |  |  |
| 6 | qa_1e61ddf3a44c0ea91ba9 | wrong_span_same_sentence | Kapan Red Hot Chili Peppers dibentuk? | 1983 | Red Hot Chili Peppers (lebih dikenal | time |  |  |
| 7 | qa_120d3e2aeafdd83b1379 | false_no_answer | Kapan Minecraft pertama kali diciptakan? | 17 Mei 2011 | (no answer) | time |  |  |
| 8 | qa_05472ac12e5fdcef4ee0 | wrong_sentence | Dimana letak Düsseldorf? | Jerman | Düsseldorf terletak di tepi Sungai Rhein | location |  |  |
| 9 | qa_24041b11d835d85aa0ca | boundary_too_long | Apa nama bandara pertama di Indonesia? | Bandar Udara Kemayoran | Bandar Udara Kemayoran merupakan bandar udara pertama di Indonesia yang dibuka untuk penerbangan | entity_or_def |  |  |
| 10 | qa_1e53abdcb86f5d50175c | partial_overlap | Apakah yang dimaksud Gelombang gravitasi? | riak dalam lengkung ruang-waktu yang bergerak dalam bentuk gelombang menjauhi sumbernya | gelombang gravitasi adalah riak dalam lengkung ruang-waktu yang bergerak dalam bent | entity_or_def |  |  |
| 11 | qa_1381e371016dcc40f018 | false_no_answer | Kapan layar sentuh pada handphone pertama kali diciptakan ? | 1992 | (no answer) | time |  |  |
| 12 | qa_205eb0af618660bf8d4e | wrong_sentence | Apakah nama ibukota Kalimantan Timur? | Samarinda | Kalimantan Timur (disingkat Kaltim) adalah | entity_or_def |  |  |
| 13 | qa_0674f81bcf68276feae4 | boundary_too_long | Apakah yang dimaksud dengan algoritme? | prosedur langkah-demi-langkah untuk penghitungan | komputer, algoritme adalah prosedur langkah-demi-langkah untuk penghitungan. Algoritme digunakan | entity_or_def |  |  |
| 14 | qa_140ea77136a742b49cea | false_no_answer | berapakah jumlah jenis fauna yang ada di Indonesia? | 147 spesies mamalia, 114 burung, 91 ikan dan 2 invertebrata | (no answer) | number |  |  |
| 15 | qa_180f09da0a4c0a1fd302 | wrong_sentence | Apa warna asli Angsa batu kaki biru? | coklat | angsa batu kaki biru menunjukkan warna | entity_or_def |  |  |
| 16 | qa_1517da00eb83219c0d8e | boundary_too_long | Apakah rumah adat Maluku? | Rumah Baileo | adalah rumah adat Maluku dan Maluku Utara, Indonesia.[1] Rumah Baileo merupakan representasi kebudayaan Maluku | entity_or_def |  |  |
| 17 | qa_08d2b4d14fb91aadc8be | false_no_answer | Kapan Vladimir Sviatoslavich pindah agama menjadi Kristen ? | 988 | (no answer) | time |  |  |
| 18 | qa_117ffbb8cd4ed821d9e9 | boundary_too_long | apakah nama ibukota Vietnam? | Hanoi | Ibukota Vietnam adalah Hanoi | entity_or_def |  |  |
| 19 | qa_20564c45de2060da8fb1 | false_no_answer | Kapan Yesus lahir ? | 6 SM | (no answer) | time |  |  |
| 20 | qa_0d3166a8ccb592c32956 | boundary_too_long | Apa nama suku asli masyarakat Banyuwangi ? | Suku Using | Suku Using adalah penduduk asli Banyuwangi | entity_or_def |  |  |
| 21 | qa_01364a9092f96eb6e2be | false_no_answer | Apakah kota terbesar di Iran? | Teheran | (no answer) | entity_or_def |  |  |
| 22 | qa_1f4e980b4aba401ebe8e | boundary_too_long | berapakah luas Papua? | 808.105km persegi | Papua memiliki luas 808.105km persegi | number |  |  |
| 23 | qa_214ca9b6b63b3090911d | false_no_answer | Apa penyebab utama Penyakit pulmonari obstruktif kronis? | Merokok tembakau | (no answer) | entity_or_def |  |  |
| 24 | qa_0a266524498967311c7b | boundary_too_long | apakah ibukota provinsi  Kalimantan Selatan? | Banjarmasin | adalah Banjarmasin. Provinsi Kalimantan Selatan | entity_or_def |  |  |
| 25 | qa_09df0b3461ac400e9981 | false_no_answer | kapankah Yesus Kristus dilahirkan? | sesudah tahun 6 SM | (no answer) | time |  |  |
| 26 | qa_14b925e5c6d46c010913 | boundary_too_long | negara manakah yang mempunyai jaringan Internet pertama kali? | Amerika Serikat | Departemen Pertahanan Amerika Serikat | location |  |  |
| 27 | qa_14c02949ccea2ed9dabd | false_no_answer | Kapan Kinetoskop diciptakan ? | 1889 / 1889 dan 1892 | (no answer) | time |  |  |
| 28 | qa_244f531e28dd211e2fb1 | false_no_answer | Siapakah tokoh antagonis dalam sinetron Mutiara Hati ? | Marchia Caroline Pohan | (no answer) | person |  |  |
| 29 | qa_032e963768b50c42794c | false_no_answer | Apakah ibu kota Tanzania? | Dar-es-Salam | (no answer) | entity_or_def |  |  |
| 30 | qa_233a408f80fd80b4c302 | false_no_answer | Apakah nama ibukota Jepang? | Tokyo | (no answer) | entity_or_def |  |  |
| 31 | qa_15fcf4146498cca19dcf | false_no_answer | Siapakah yang tidak sedikit memberikan kontribusi bagi berdirinya Dinasti Tang? | Li Jiancheng | (no answer) | person |  |  |
| 32 | qa_1aaf53e8bdeae8febcd3 | false_no_answer | Siapa penemu Aljabar? | Al-Khwārizmī | (no answer) | person |  |  |
| 33 | qa_1c8a15cd9ed413c6ac4b | false_no_answer | Kapan Masjid Shah  dibangun ? | 1611 | (no answer) | time |  |  |
| 34 | qa_1a05439cf9633a04e7eb | false_no_answer | apakah nama ibukota Polandia ? | Warsawa | (no answer) | entity_or_def |  |  |
| 35 | qa_215981b00544ba9a3c1c | false_no_answer | Apakah kesenian tradisional khas Gunungkidul yang mencerminkan bahwa kehidupan manusia tidak jauh dari alam? | Rinding Gumbeng | (no answer) | entity_or_def |  |  |
| 36 | qa_175e4e7028a6638527bf | false_no_answer | dimanakah letak gunung Fuji di Jepang ? | perbatasan Prefektur Shizuoka dan Yamanashi, di sebelah barat Tokyo | (no answer) | location |  |  |
| 37 | qa_04b38b92e0a507d36cd7 | false_no_answer | Siapakah Menteri Transmigrasi dan Pemukiman Proyek lahan gambut satu juta hektar? | Siswono Yudo Husodo | (no answer) | person |  |  |
| 38 | qa_195a889e959cd03a8d8f | false_no_answer | Apa nama brand handphone pertama yang menggunakan sistem layar sentuh ? | Simon | (no answer) | entity_or_def |  |  |
| 39 | qa_24919d8a11cb4ba5d678 | false_no_answer | Siapa yang pertama kali menjual telepon cerdas? | BellSouth | (no answer) | person |  |  |
| 40 | qa_00e198cdbe7a6db9fff7 | false_no_answer | Kapan Yan Xishan diangkat menjadi panglima perang Tiongkok ? | 1911 | (no answer) | time |  |  |
| 41 | qa_0bf9cafef11db24ae688 | correct_low_conf | Siapa yang menemukan Tabel periodik? | Dmitri Mendeleev | Dmitri Mendeleev | person |  |  |
| 42 | qa_0986961735e0c79ff572 | correct_low_conf | Siapakah nama pemimpin dalam gereja ? | Paus | Paus | person |  |  |
| 43 | qa_11f43b983ed2d07230d3 | correct_low_conf | apakah nama ibukota Israel? | Tel Aviv | Tel Aviv | entity_or_def |  |  |
| 44 | qa_1e0375bb9d9e7cb9488b | correct_low_conf | Apa ibukota Israel ? | Tel Aviv | Tel Aviv | entity_or_def |  |  |
