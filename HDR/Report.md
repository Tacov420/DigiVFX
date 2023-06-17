# VFX HW1 Report

##### B09902043 沈竑文 B09902040 洪郁凱

- 描述作業內容
  - 讀取多張相近時間下，相同地點、視角、相同拍攝物體但不同曝光時間長度的照片
  
  - 利用曝光時間長的照片捕捉暗處細節
  
  - 利用曝光時間短的照片紀錄亮處細節
  
  - 合成出一張接近人類inception的照片
  
    
  
- 實作演算法
  - 實作MTB alignment (加分項目)
  
  - 將多張照片合成一張HDR照片
  
  - 完成Tone mapping (Reinhard) (加分項目)
  
    
  
- 實作細節
  - MTB alignment
    - 讀入多張彩色照片
    
    - 將每張彩色照片取G通道的亮度當作灰階照片
    
    - 根據每張灰階照片的中位數，生出Median Threshold bitmap
    
    - 分別將第$[2,n]$張照片向第一張照片對齊
      - 將兩張需對齊的照片縮到$2^d$倍小
      - 將後者預先位移已經累積的$(x,y)$ 
      - 將後者對x/y軸分別位移$[-1,1]$共九次，取bitwise XOR得出每個位移的error值
      - 取error值最小的$(dx,dy)$加入累計的$(x,y)$中
      - 將(x,y)分別乘以二，並將d減一，一直操作直到d=0
      
    - 得到每張圖片的位移量之後，根據位移輸出對齊後的照片
    
      
    
  - HDR組合 (使用Debevec的方法)
    - 讀入$p$張對齊好的彩色照片
    
    - 對每張照片的3個顏色通道選$n$個點，$np>p+256$，共$3np$個強度值(0-255)
    
    - 生出定義域為$[0,255]$的Weight function(參考課程投影片)
    
    - 利用選定的多個點在照片的值、個別照片的曝光時間、$g(Z_{mid})$以及對$g$函數要smooth的假設，得出$np+1+254$條方程式，將代表那些方程式的矩陣內對應位置的值填好後解least-square solution來得到$g$函數
    
    - 利用g函數及多張照片，加權平均後估計出真實場景的每個點的radiance值
    
      
    
  - Reinhard tone mapping (global)
    - 將radiance的RGB經由不同權數的加權平均(約$2:7:1$)，得到$L_w$
    
    - 參考課程投影片公式生出$L_w^{avg}$，並用其得到$L_m$
    
    - 根據$L_m$，生出每個pixel的$L_d$值
    
    - 最後根據$R_w、L_w、L_d$得到$R_d$，而$G_d、B_d$則同理，將最後結果存為`result.jpg`
    
      
    
  - Adaptive Logarithmic Mapping
  
    * [Ref](https://resources.mpi-inf.mpg.de/tmo/logmap/logmap.pdf)
  
    * 先將RGB的Radiance透過矩陣乘法轉換成CIE XYZ色彩空間
    
      - [Ref](https://www.oceanopticsbook.info/view/photometry-and-visibility/from-xyz-to-rgb)
      - 其中，CIE[1]即為$L_w$
    
    * 根據Paper中的公式4，我們便能得到每個Pixel的$L_d$
    
    * 利用$L_d$，建構出新的CIE XYZ色彩空間
    
    * 參考Paper中的Gamma Correction的部分，修正一下
    
    * 最後將新的CIE XYZ色彩空間的值換回RGB即可得到`result.jpg`
    
      
  
- 實作結果 (含有test images以及我們拍的照片)
  * 不同lambda (先固定使用Reinhard tone mapping，其中gamma = $0.1$, key = $0.8$ 方便討論)
  
    | lambda | function g                                                   | result image                                                 |
    | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | 1      | <img src="D:\DigiVFX\HW 1\Figure_1.png" style="zoom:72%;" /> | <img src="D:\DigiVFX\HW 1\desk_l1.jpg" style="zoom:50%;" />  |
    | 16     | <img src="D:\DigiVFX\HW 1\Figure_16.png" style="zoom:72%;" /> | <img src="D:\DigiVFX\HW 1\desk_l16.jpg" style="zoom:50%;" /> |
    | 40     | <img src="D:\DigiVFX\HW 1\Figure_1_40.png" style="zoom:72%;" /> | <img src="D:\DigiVFX\HW 1\desk_l40.jpg" style="zoom:50%;" /> |
  
    很明顯，lambda的提高的確能幫助找出的$g$函數更加平滑，但結果品質我覺得差不多
  
    
  
  * 不同照片組 (Reinhard，固定$lambda = 40, gamma=0.2, key = 0.8$)
  
    |             | function g                                                   | result                                                     |
    | ----------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
    | Test images | <img src="D:\DigiVFX\HW 1\Figure_1_40.png" style="zoom:72%;" /> | <img src="D:\DigiVFX\HW 1\desk.jpg" style="zoom:50%;" />   |
    | Our images  | <img src="D:\DigiVFX\HW 1\Figure_2.png" style="zoom:72%;" /> | <img src="D:\DigiVFX\HW 1\wojak.jpg" style="zoom: 13%;" /> |
  
    觀察是，不同組照片因為照片本身或拍攝設備性質不同，解出來的$g$可能會有些許差異，單純使用Reinhard global operator的效果就不一定會那麼好。
  
    此外，我們的圖片中也能更明顯觀察到使用Reinhard tone mapping後的結果色調會偏橘黃。
  
    
  
  * 不同Tone Mapping方式
  
    |             | Reinhard Global Operator                                  | Adaptive Logarithmic Mapping                                 |
    | ----------- | --------------------------------------------------------- | ------------------------------------------------------------ |
    | Test images | <img src="D:\DigiVFX\HW 1\desk.jpg" style="zoom:50%;" />  | <img src="D:\DigiVFX\HW 1\result_log.jpg" style="zoom:50%;" /> |
    | Our images  | <img src="D:\DigiVFX\HW 1\wojak.jpg" style="zoom:13%;" /> | <img src="C:\Users\User\Downloads\result_log_1.25.jpg" style="zoom:13%;" /> |
  
    可以發現Adaptive Logarithmic Mapping的結果就不會那麼偏向橘黃色調，甚至在我們照片中暗部資訊更明顯。

