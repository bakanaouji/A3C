# 論文メモ書き

## Momentum SGD
非同期なSGDの実装は"A lock-free approach to paralleliz- ing stochastic gradient descent"などで検討されていて，実装が簡単．

$$\theta$$を全てのスレッド間で共有するパラメータベクトル，$$\Delta \theta_i$$をi番目のスレッドによって計算された$$\theta$$の勾配とする．

各スレッドiはモメンタム項の更新$$m_i=\alpha m_i+(1-\alpha)\Delta\theta_i$$とパラメータの更新$$\theta\gets\theta-\eta m_i$$をロックなしで独立して行う．

ここで，各スレッドは独自の勾配とモメンタムベクトルを保持している．