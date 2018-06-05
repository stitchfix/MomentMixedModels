package com.stitchfix.mbest

import breeze.linalg.{DenseMatrix, DenseVector}
import com.stitchfix.mbest.MatrixUtils.{CompactSVD, compactSVD}
import com.stitchfix.mbest.MixedEffectsRegression.{Data, Estimate_η0}

trait SleepStudyTestData {
  val m: DenseMatrix[Double] = DenseMatrix(
    (1.0, 0.0),
    (1.0, 1.0),
    (1.0, 2.0),
    (1.0, 3.0),
    (1.0, 4.0),
    (1.0, 5.0),
    (1.0, 6.0),
    (1.0, 7.0),
    (1.0, 8.0),
    (1.0, 9.0)
  )

  val y308: DenseVector[Double] = DenseVector(249.5600, 258.7047, 250.8006, 321.4398, 356.8519, 414.6901, 382.2038, 290.1486, 430.5853, 466.3535)
  val y309: DenseVector[Double] = DenseVector(222.7339, 205.2658, 202.9778, 204.7070, 207.7161, 215.9618, 213.6303, 217.7272, 224.2957, 237.3142)
  val y310: DenseVector[Double] = DenseVector(199.0539, 194.3322, 234.3200, 232.8416, 229.3074, 220.4579, 235.4208, 255.7511, 261.0125, 247.5153)
  val y330: DenseVector[Double] = DenseVector(321.5426, 300.4002, 283.8565, 285.1330, 285.7973, 297.5855, 280.2396, 318.2613, 305.3495, 354.0487)
  val y331: DenseVector[Double] = DenseVector(287.6079, 285.0000, 301.8206, 320.1153, 316.2773, 293.3187, 290.0750, 334.8177, 293.7469, 371.5811)
  val y332: DenseVector[Double] = DenseVector(234.8606, 242.8118, 272.9613, 309.7688, 317.4629, 309.9976, 454.1619, 346.8311, 330.3003, 253.8644)
  val y333: DenseVector[Double] = DenseVector(283.8424, 289.5550, 276.7693, 299.8097, 297.1710, 338.1665, 332.0265, 348.8399, 333.3600, 362.0428)
  val y334: DenseVector[Double] = DenseVector(265.4731, 276.2012, 243.3647, 254.6723, 279.0244, 284.1912, 305.5248, 331.5229, 335.7469, 377.2990)
  val y335: DenseVector[Double] = DenseVector(241.6083, 273.9472, 254.4907, 270.8021, 251.4519, 254.6362, 245.4523, 235.3110, 235.7541, 237.2466)
  val y337: DenseVector[Double] = DenseVector(312.3666, 313.8058, 291.6112, 346.1222, 365.7324, 391.8385, 404.2601, 416.6923, 455.8643, 458.9167)
  val y349: DenseVector[Double] = DenseVector(236.1032, 230.3167, 238.9256, 254.9220, 250.7103, 269.7744, 281.5648, 308.1020, 336.2806, 351.6451)
  val y350: DenseVector[Double] = DenseVector(256.2968, 243.4543, 256.2046, 255.5271, 268.9165, 329.7247, 379.4445, 362.9184, 394.4872, 389.0527)
  val y351: DenseVector[Double] = DenseVector(250.5265, 300.0576, 269.8939, 280.5891, 271.8274, 304.6336, 287.7466, 266.5955, 321.5418, 347.5655)
  val y352: DenseVector[Double] = DenseVector(221.6771, 298.1939, 326.8785, 346.8555, 348.7402, 352.8287, 354.4266, 360.4326, 375.6406, 388.5417)
  val y369: DenseVector[Double] = DenseVector(271.9235, 268.4369, 257.2424, 277.6566, 314.8222, 317.2135, 298.1353, 348.1229, 340.2800, 366.5131)
  val y370: DenseVector[Double] = DenseVector(225.2640, 234.5235, 238.9008, 240.4730, 267.5373, 344.1937, 281.1481, 347.5855, 365.1630, 372.2288)
  val y371: DenseVector[Double] = DenseVector(269.8804, 272.4428, 277.8989, 281.7895, 279.1705, 284.5120, 259.2658, 304.6306, 350.7807, 369.4692)
  val y372: DenseVector[Double] = DenseVector(269.4117, 273.4740, 297.5968, 310.6316, 287.1726, 329.6076, 334.4818, 343.2199, 369.1417, 364.1236)

  val data308: Data = Data(m, m, y308)
  val data309: Data = Data(m, m, y309)
  val data310: Data = Data(m, m, y310)
  val data330: Data = Data(m, m, y330)
  val data331: Data = Data(m, m, y331)
  val data332: Data = Data(m, m, y332)
  val data333: Data = Data(m, m, y333)
  val data334: Data = Data(m, m, y334)
  val data335: Data = Data(m, m, y335)
  val data337: Data = Data(m, m, y337)
  val data349: Data = Data(m, m, y349)
  val data350: Data = Data(m, m, y350)
  val data351: Data = Data(m, m, y351)
  val data352: Data = Data(m, m, y352)
  val data369: Data = Data(m, m, y369)
  val data370: Data = Data(m, m, y370)
  val data371: Data = Data(m, m, y371)
  val data372: Data = Data(m, m, y372)

  val csvd: CompactSVD = compactSVD(data308.F, 2)
  val sleepStudy_η0_Σ: DenseMatrix[Double] = DenseMatrix((0.001711295, 0.0), (0.0,  0.177076584))

  val η0_308: Estimate_η0 = Estimate_η0(DenseVector(-42.39086, 168.0919), sleepStudy_η0_Σ)
  val η0_309: Estimate_η0 = Estimate_η0(DenseVector(-24.41398, 142.9345), sleepStudy_η0_Σ)
  val η0_310: Estimate_η0 = Estimate_η0(DenseVector(-26.92963, 141.4086), sleepStudy_η0_Σ)
  val η0_330: Estimate_η0 = Estimate_η0(DenseVector(-34.35939, 201.9472), sleepStudy_η0_Σ)
  val η0_331: Estimate_η0 = Estimate_η0(DenseVector(-35.49664, 198.9403), sleepStudy_η0_Σ)
  val η0_332: Estimate_η0 = Estimate_η0(DenseVector(-36.10698, 183.4571), sleepStudy_η0_Σ)
  val η0_333: Estimate_η0 = Estimate_η0(DenseVector(-37.00945, 191.0231), sleepStudy_η0_Σ)
  val η0_334: Estimate_η0 = Estimate_η0(DenseVector(-35.30035, 166.3372), sleepStudy_η0_Σ)
  val η0_335: Estimate_η0 = Estimate_η0(DenseVector(-27.27937, 183.9935), sleepStudy_η0_Σ)
  val η0_337: Estimate_η0 = Estimate_η0(DenseVector(-45.59108, 200.4561), sleepStudy_η0_Σ)
  val η0_349: Estimate_η0 = Estimate_η0(DenseVector(-33.37711, 148.7063), sleepStudy_η0_Σ)
  val η0_350: Estimate_η0 = Estimate_η0(DenseVector(-38.76793, 155.5245), sleepStudy_η0_Σ)
  val η0_351: Estimate_η0 = Estimate_η0(DenseVector(-33.57334, 181.6381), sleepStudy_η0_Σ)
  val η0_352: Estimate_η0 = Estimate_η0(DenseVector(-40.24967, 191.4752), sleepStudy_η0_Σ)
  val η0_369: Estimate_η0 = Estimate_η0(DenseVector(-36.31707, 176.7762), sleepStudy_η0_Σ)
  val η0_370: Estimate_η0 = Estimate_η0(DenseVector(-36.04360, 144.9423), sleepStudy_η0_Σ)
  val η0_371: Estimate_η0 = Estimate_η0(DenseVector(-34.66067, 176.0865), sleepStudy_η0_Σ)
  val η0_372: Estimate_η0 = Estimate_η0(DenseVector(-37.62696, 185.2147), sleepStudy_η0_Σ)

  val sleepStudyInputData = Seq(
    ("308", (data308, csvd, η0_308)),
    ("309", (data309, csvd, η0_309)),
    ("310", (data310, csvd, η0_310)),
    ("330", (data330, csvd, η0_330)),
    ("331", (data331, csvd, η0_331)),
    ("332", (data332, csvd, η0_332)),
    ("333", (data333, csvd, η0_333)),
    ("334", (data334, csvd, η0_334)),
    ("335", (data335, csvd, η0_335)),
    ("337", (data337, csvd, η0_337)),
    ("349", (data349, csvd, η0_349)),
    ("350", (data350, csvd, η0_350)),
    ("351", (data351, csvd, η0_351)),
    ("352", (data352, csvd, η0_352)),
    ("369", (data369, csvd, η0_369)),
    ("370", (data370, csvd, η0_370)),
    ("371", (data371, csvd, η0_371)),
    ("372", (data372, csvd, η0_372))
  )
}
