# Error sample: qwen0.5b_en1_0shot_n100 (qwen0.5b_en1_0shot_n100)

40 errors stratified by outcome (round robin) + 10 correct answers with the highest na_prob. Fill human_label with one of: gold_issue, multi_hop_miss, right_type_wrong_entity, wrong_type, none, other.

| # | id | outcome | question | gold | pred | q_type | human_label | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | qa_1cafbf04e5059f20d957 | false_no_answer | siapakah karakter uatama anime One Piece? | Monkey D. Luffy | (no answer) | person |  |  |
| 2 | qa_205eb0af618660bf8d4e | wrong_sentence | Apakah nama ibukota Kalimantan Timur? | Samarinda | Kalimantan Timur (disingkat Kaltim) adalah | entity_or_def |  |  |
| 3 | qa_05aed54e82bdb9edebf5 | boundary_too_long | Apakah ibukota Papua Nugini? | Port Moresby | antaranya ibukota Port Moresby | entity_or_def |  |  |
| 4 | qa_1e53abdcb86f5d50175c | partial_overlap | Apakah yang dimaksud Gelombang gravitasi? | riak dalam lengkung ruang-waktu yang bergerak dalam bentuk gelombang menjauhi sumbernya | gelombang gravitasi adalah riak dalam lengkung ruang-waktu yang bergerak dalam bent | entity_or_def |  |  |
| 5 | qa_0d82790c6d3244970680 | boundary_too_short | siapakah orang tua Mahasthavira Vajragiri? | Ayahnya bernama The Tjai Gwan dan ibunya Liem Kiat Nio | The Tjai Gwan dan ibunya Liem Kiat Nio | person |  |  |
| 6 | qa_0fd14e04b7219bdf46fc | false_answer_on_unanswerable | Di mana gitaris Boys Like Girls? | (no answer) | Boys Like Girls pertama dibentuk di | location |  |  |
| 7 | qa_233a408f80fd80b4c302 | wrong_span_same_sentence | Apakah nama ibukota Jepang? | Tokyo | Ibu kota Jepang | entity_or_def |  |  |
| 8 | qa_1e61ddf3a44c0ea91ba9 | false_no_answer | Kapan Red Hot Chili Peppers dibentuk? | 1983 | (no answer) | time |  |  |
| 9 | qa_1a05439cf9633a04e7eb | wrong_sentence | apakah nama ibukota Polandia ? | Warsawa | Polandia adalah | entity_or_def |  |  |
| 10 | qa_1f4e980b4aba401ebe8e | boundary_too_long | berapakah luas Papua? | 808.105km persegi | Papua memiliki luas 808.105km persegi | number |  |  |
| 11 | qa_074634b2eae86659178f | partial_overlap | Berapa luas Pulau Serangan? | panjang maksium 2,9 km dan lebar 1 km | Luas Pulau Serangan adalah 2,9 km persegi | number |  |  |
| 12 | qa_1737a489cc79e73c20c9 | false_answer_on_unanswerable | Di manakah jerut jerang berasal? | (no answer) | berasal dari Asia Timur, sedangkan jeruk | location |  |  |
| 13 | qa_01364a9092f96eb6e2be | false_no_answer | Apakah kota terbesar di Iran? | Teheran | (no answer) | entity_or_def |  |  |
| 14 | qa_180f09da0a4c0a1fd302 | wrong_sentence | Apa warna asli Angsa batu kaki biru? | coklat | angsa batu kaki biru menunjukkan warna | entity_or_def |  |  |
| 15 | qa_175e4e7028a6638527bf | boundary_too_long | dimanakah letak gunung Fuji di Jepang ? | perbatasan Prefektur Shizuoka dan Yamanashi, di sebelah barat Tokyo | terletak di perbatasan Prefektur Shizuoka dan Yamanashi, di sebelah barat Tokyo. Gunung Fuji | location |  |  |
| 16 | qa_1cf7b4abf96720c516d8 | false_answer_on_unanswerable | Siapa yang menciptakan tabel non-elektromagnetik? | (no answer) | Dmitri Mendeleev | person |  |  |
| 17 | qa_14c02949ccea2ed9dabd | false_no_answer | Kapan Kinetoskop diciptakan ? | 1889 / 1889 dan 1892 | (no answer) | time |  |  |
| 18 | qa_05472ac12e5fdcef4ee0 | wrong_sentence | Dimana letak Düsseldorf? | Jerman | Düsseldorf terletak di tepi Sungai Rhein | location |  |  |
| 19 | qa_117ffbb8cd4ed821d9e9 | boundary_too_long | apakah nama ibukota Vietnam? | Hanoi | Ibukota Vietnam adalah Hanoi | entity_or_def |  |  |
| 20 | qa_24041b11d835d85aa0ca | false_no_answer | Apa nama bandara pertama di Indonesia? | Bandar Udara Kemayoran | (no answer) | entity_or_def |  |  |
| 21 | qa_0a3867b0e2c4601ee016 | boundary_too_long | Berapa luas Papua? | 808.105km | Papua memiliki luas 808.105km persegi | number |  |  |
| 22 | qa_2235e9532c952b3717f9 | false_no_answer | Komponen apakah yang tidak pernah gagal membunuh bakteri gram positif dan bakteri gram negatif? | Peptida antimikrobial | (no answer) | entity_or_def |  |  |
| 23 | qa_1517da00eb83219c0d8e | boundary_too_long | Apakah rumah adat Maluku? | Rumah Baileo | Rumah Baileo adalah rumah adat Maluku dan Maluku Utara, Indonesia | entity_or_def |  |  |
| 24 | qa_12a96c12a171ddb1849c | false_no_answer | Apa yang tidak ada di kereta ekonomi sebelum tahun 2010? | AC | (no answer) | entity_or_def |  |  |
| 25 | qa_0a266524498967311c7b | boundary_too_long | apakah ibukota provinsi  Kalimantan Selatan? | Banjarmasin | adalah Banjarmasin. Provinsi Kalimantan Selatan | entity_or_def |  |  |
| 26 | qa_214ca9b6b63b3090911d | false_no_answer | Apa penyebab utama Penyakit pulmonari obstruktif kronis? | Merokok tembakau | (no answer) | entity_or_def |  |  |
| 27 | qa_0674f81bcf68276feae4 | boundary_too_long | Apakah yang dimaksud dengan algoritme? | prosedur langkah-demi-langkah untuk penghitungan | komputer, algoritme adalah prosedur langkah-demi-langkah untuk penghitungan. Algoritme digunakan | entity_or_def |  |  |
| 28 | qa_1e4ba2c2a06dde242174 | false_no_answer | Kapan kekaisaran Romawi Timur runtuh? | 29 Mei 1453 | (no answer) | time |  |  |
| 29 | qa_0d3166a8ccb592c32956 | boundary_too_long | Apa nama suku asli masyarakat Banyuwangi ? | Suku Using | Suku Osing atau biasa diucapkan Suku Using | entity_or_def |  |  |
| 30 | qa_2548a969bbcdbeb6bc0d | false_no_answer | Kapan telegram pertama ditemukan? | 1837 | (no answer) | time |  |  |
| 31 | qa_09df0b3461ac400e9981 | false_no_answer | kapankah Yesus Kristus dilahirkan? | sesudah tahun 6 SM | (no answer) | time |  |  |
| 32 | qa_14b925e5c6d46c010913 | false_no_answer | negara manakah yang mempunyai jaringan Internet pertama kali? | Amerika Serikat | (no answer) | location |  |  |
| 33 | qa_120d3e2aeafdd83b1379 | false_no_answer | Kapan Minecraft pertama kali diciptakan? | 17 Mei 2011 | (no answer) | time |  |  |
| 34 | qa_16a2febc8c0b6aa56cea | false_no_answer | berapakah jumlah kecamatan di Kalimantan Selatan? | 153 | (no answer) | number |  |  |
| 35 | qa_04b38b92e0a507d36cd7 | false_no_answer | Siapakah Menteri Transmigrasi dan Pemukiman Proyek lahan gambut satu juta hektar? | Siswono Yudo Husodo | (no answer) | person |  |  |
| 36 | qa_244f531e28dd211e2fb1 | false_no_answer | Siapakah tokoh antagonis dalam sinetron Mutiara Hati ? | Marchia Caroline Pohan | (no answer) | person |  |  |
| 37 | qa_0349eeca8eb4cae07506 | false_no_answer | kapankah papua nugini melepaskan diri dari indonesia? | 1 Juli 1971 | (no answer) | time |  |  |
| 38 | qa_20564c45de2060da8fb1 | false_no_answer | Kapan Yesus lahir ? | 6 SM | (no answer) | time |  |  |
| 39 | qa_17e01bfa1eeedaf3a2be | false_no_answer | Kapan ilmu astronomi diperkenalkan ? | abad ke-3 SM | (no answer) | time |  |  |
| 40 | qa_032e963768b50c42794c | false_no_answer | Apakah ibu kota Tanzania? | Dar-es-Salam | (no answer) | entity_or_def |  |  |
| 41 | qa_1e0375bb9d9e7cb9488b | correct_low_conf | Apa ibukota Israel ? | Tel Aviv | Tel Aviv | entity_or_def |  |  |
| 42 | qa_0986961735e0c79ff572 | correct_low_conf | Siapakah nama pemimpin dalam gereja ? | Paus | Paus | person |  |  |
| 43 | qa_11f43b983ed2d07230d3 | correct_low_conf | apakah nama ibukota Israel? | Tel Aviv | Tel Aviv | entity_or_def |  |  |
| 44 | qa_0bf9cafef11db24ae688 | correct_low_conf | Siapa yang menemukan Tabel periodik? | Dmitri Mendeleev | Dmitri Mendeleev | person |  |  |
