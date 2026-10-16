<!-- Parent: ../../AGENTS.md -->

# Native 퍼즐의 packing과 전이

이 디렉터리는 native 퍼즐과 Random/Hard 변형, Room, Cayley 파생 클래스의 상태 표현을 소유한다.
PDDL과 world-model 표현은 별도 소유자다. 퍼즐의 논리값, action 번호, 비용, 역전이 의미를
유지하면서 저장 표현을 최적화한다. 기존 논리 입력과 DeepCubeA/Santa 데이터는 유지한다.
내부 State의 과거 packed 바이트나 직렬화된 PyTree는 안정적인 외부 포맷으로 취급하지 않는다.

## 용량을 계산하는 방법

- **정보 하한**: 유효 상태가 `S`개면 `ceil(log2(S))`비트다. 이를 달성하는 rank/unrank가
  빠른 전이와 부분 접근을 제공한다는 뜻은 아니다.
- **고정폭 원소 하한**: 값이 `0..M`이면 `max(1, M.bit_length())`비트다. 값의 개수와
  최댓값을 혼동하지 않는다. `1..N` 순열은 `N.bit_length()`가 필요하다.
- **실제 State 저장량**: 저장 PyTree leaf들의 `nbytes` 합이다. `_unpacked` view의 크기,
  GPU allocator의 예약량, 전체 탐색 자료구조 용량과 구분한다.
- **공유 문맥**: Maze 벽과 TSP 좌표·거리 행렬은 `SolveConfig.InstanceContext`에 있다.
  단일 문제를 탐색할 때 상태 수만큼 중복 저장하지 않는다. 서로 다른 문제를 배치하면 문맥도 배치된다.
- **임시 메모리와 통신량**: unpack, gather/scatter, repack의 중간 배열과 출력 크기를 함께 본다.
  저장 바이트 감소가 메모리 트랜잭션·커널 수·실행 시간 감소를 보장하지 않는다.

`FieldDescriptor.packed_tensor(shape=..., packed_bits=b)`는 필드 자체를 packed `uint8`로
저장한다. 논리 원소 수를 `N`, `L=lcm(b,8)`이라고 하면 현재 codec의 저장량은
`ceil(N / (L/b)) * (L/8)`바이트다. 예를 들어 25개 × 5비트는 이상적인 16바이트가 아니라
20바이트이고, 좌표 두 개 × 5비트는 5바이트다. 둘 다 블록 끝 padding 때문이다.

일반 `tensor(..., bits=b)`의 비트 메타데이터만 바꿔도 기본 State leaf가 작아지는 것은 아니다.
aggregate `.packed` 표현을 실제 저장소에 사용해야 한다. aggregate codec은 32비트 word와
작은 tail을 사용하며 3바이트 tail은 4바이트 word를 소비한다. 따라서 field packing보다
항상 작지도 않다. 35개 × 6비트는 field 27바이트, aggregate 28바이트다.

## 표현과 전이의 계약

1. 논리 입력은 `State.from_unpacked(...)`로, 논리 필드 수정은 `set_unpacked(...)`로 처리한다.
   raw 생성자와 `replace`에는 저장 dtype·shape에 맞는 값을 전달한다.
2. packed 1비트 필드는 little-bit order다. 셀 `i`는 바이트 `i//8`의 비트 `i%8`이다.
   사용하지 않는 마지막 비트는 0을 유지한다. 같은 논리 상태의 padding이 다르면 저장 바이트 기반
   equality/hash가 달라질 수 있다.
3. packed 값 위에서 XOR, 단일 비트 검사·갱신이 가능하면 전체 unpack/repack을 피한다.
   반대로 permutation의 불규칙 gather에 좁은 비트를 적용하면 decode 비용이 더 클 수 있다.
4. 좁은 dtype은 저장용이다. 좌표 차·음수 이동, 행렬 곱의 중간 연산은 넓은 signed dtype으로
   수행한 뒤 저장 dtype으로 되돌린다. 유효성 검사를 narrowing 뒤에 수행하지 않는다.
5. 생성 시 dtype을 줄일 때 난수 생성의 기존 dtype을 유지하고 결과만 변환하면 동일 seed의
   논리 입력을 보존할 수 있다. dtype을 바꿔 `random.randint`를 호출하면 표본도 바뀔 수 있다.
6. `_apply`는 단일 action의 후보 상태와 비용, `_get_neighbours`는 모든 action의 후보 상태와
   비용 벡터를 반환한다. `filled=False` 또는 유한하지 않은 비용일 때 원래 상태로 복구하는 책임은
   각각 `Puzzle.get_actions`와 `Puzzle.get_neighbours`에 있다. private 커널에 이를 복제하지 않는다.
7. State뿐 아니라 GoalSpec, InstanceContext의 상태 인덱스, 역탐색 초기값, 렌더러,
   JAxtar heuristic/neural preprocess, benchmark 입력 경로를 함께 확인한다.
8. 배치 축은 논리 필드 내부 축과 다르다. 단일 상태용 bit indexing은 JAX vmap을 통해 배치한다.
   외부의 raw State 생성에 의존해 정규 dtype/shape를 추론하지 않는다.

## 단일 전이와 전체 이웃의 독립 최적화

모든 native 기본 클래스는 `_apply`와 `_get_neighbours`를 각각 구현한다. Random/Hard와
Room, Cayley 파생 클래스는 해당 커널 쌍을 상속한다. 단일 action에서 전체 이웃을 생성한 뒤
하나를 고르거나, native 전체 이웃을 `vmap(self.get_actions)`로 고정하지 않는다.
두 private 커널은 서로 호출하지 않으므로 한쪽의 인덱싱·배열 배치·전처리를 별도로 바꿀 수 있다.
파생 클래스가 전이 의미를 바꾸면 두 커널을 함께 구현한다. 목표나 초기 표본만 바꾸는 파생
클래스는 두 커널을 그대로 상속할 수 있다. 한 커널의 override가 다른 경로에 자동 전파되지는 않는다.

두 경로는 같은 action 번호에서 같은 논리 상태와 비용을 반환해야 한다. `_get_neighbours`의
선행 축은 action 순서이며, public batched API의 출력은 `(action, batch, ...)`다.
`_apply`의 입력 action은 `0 <= action < action_size`인 scalar JAX 배열이다. public 경계에서
Python 정수도 정규화한다. 무효 전이의 최종 상태에는 입력 상태의 모든 leaf가 그대로 남아야 한다.

공통 `Puzzle`은 bulk hook이 없는 환경의 기존 public `get_actions` mapping을 유지한다.
PDDL의 cost-only filled 처리나 world-model의 자체 batch 구현을 native 계약으로 강제 변환하지 않는다.
native 최적화 때문에 범위 밖 환경의 의미를 바꾸지 않는다.

| 퍼즐 | 단일 action 경로 | 전체 neighbor 경로 |
| --- | --- | --- |
| DotKnot | 활성 endpoint를 찾고 출발·도착 nibble만 갱신 | endpoint 선택을 공유하고 네 방향의 nibble mask를 생성 |
| Hanoi | source/target의 최소 disk를 찾고 peg 값 하나를 변경 | 모든 peg의 top disk를 공유하고 기본 여섯 이동을 조합 |
| LightsOut | 선택한 action의 packed toggle mask 한 행만 조회해 XOR | 상태와 무관한 전체 toggle mask를 미리 pack한 뒤 XOR |
| Maze / Room | 목적지 한 칸의 packed wall bit 조회 | 네 좌표와 네 wall bit를 벡터로 조회 |
| Pancake | 선택한 prefix의 source index를 계산해 gather | 모든 prefix의 고정 index 표로 gather |
| Rubik | 선택한 sticker permutation으로 gather | unpack을 공유하고 permutation 행별 회전과 pack을 vmap |
| Slide | 한 바이트 보드는 직접 교환, 6비트 select/pack, 8비트 index gather, 나머지는 packed XOR | blank 공유; 8비트 select/pack, 나머지는 packed XOR |
| Sokoban | player·목적지·push 위치의 2비트 필드 갱신 | player 위치를 공유하고 네 move/push를 벡터화 |
| TopSpin | 선택한 동작의 source index로 gather | 세 고정 동작을 독립 bulk 커널에서 vmap |
| TSP | 방문 bit 하나를 검사·설정하고 closing leg 판정 | 방문 정보와 마지막 방문 여부를 공유해 모든 목적지 계산 |
| Cayley | 선택한 permutation 또는 modular matrix 적용 | 전체 permutation gather 또는 batched matrix 곱 |

같은 수학적 변환이어도 `gather`, `stack`, `vmap`의 배치 축이 XLA fusion과 transpose에 영향을 준다.
한쪽 경로의 표현식을 그대로 확장하는 것으로 최적화를 끝내지 않는다. 특히 TopSpin의 전체 index
gather, 고정 출력 stack, `vmap(lax.switch)`는 forward와 inverse에서 따로 비교한다.

## 퍼즐별 분석

아래 바이트는 기본 표현의 상태당 payload다. GPU 할당량이나 탐색 전체 메모리 수치가 아니다.

| 퍼즐 | 논리 표현과 실제 저장량 | 최적성 판단 |
| --- | --- | --- |
| DotKnot | 4비트 × `size²`; 기본 10 → 50 B, 16 → 128 B | 기본 색 4개에서 원소별 4비트가 필요하다. |
| Hanoi | disk→peg 2비트; 기본 5 → 2 B, 15 → 4 B, 31 → 8 B | disk당 고정폭은 최소이며 count·빈 슬롯·정렬 중복을 제거했다. |
| LightsOut / Random | 1비트 × `size²`; 7 → 7 B, 8 → 8 B | 독립 이진 셀의 고정폭 하한을 달성한다. |
| Maze / Room | unsigned 좌표 2개; `size<=256`은 2 B, 이후 4 B | 기본 크기에서 bit codec 없이 2 B. |
| PancakeSorting | `uint8[size]`; 35 → 35 B, 64 → 64 B | 순열 정보에 비해 여유가 있으나 gather가 단순하다. |
| RubiksCube / Random | 색은 3비트 × `6*size²`; 3 → 21 B, 4 → 36 B | 색별 고정폭 하한이며 블록 padding은 남는다. |
| Rubik tile ID | 8비트 × `6*size²`; 2 → 24 B, 3 → 54 B, 4 → 96 B | 비트폭 최소보다 넓다. 불규칙 gather의 비용을 함께 본다. |
| SlidePuzzle / Random / Hard | `(size²-1).bit_length()`비트 × `size²`; 3 → 5 B, 4 → 8 B, 5 → 20 B | 원소별 폭은 최소. 크기 5에는 codec padding이 있다. |
| Sokoban / Hard | 2비트 × 100셀 → 25 B | 기본 셀 alphabet의 최소 고정폭. 정적 벽은 추가 중복이다. |
| TopSpin | `uint8[size]`; 20 → 20 B, 32 → 32 B | permutation 압축 여지가 있으나 회전·부분 반전은 byte gather가 단순하다. |
| TSP | 방문 mask `ceil(size/8)` B + point 1 또는 2 B; 16 → 3 B, 33 → 6 B, 257 → 35 B | mask는 최소 고정폭, point는 `size<=256`에서 uint8. |
| Cayley | 값 범위에 따라 원소당 1/2/4 B | graph 길이가 아니라 보존되는 값의 범위로 결정한다. |

고정된 문제 문맥에서의 정보 하한은 다음과 같이 해석한다. 이 표의 상태 수는 압축 설계의
출발점이며, 임의 논리 입력의 지원 범위를 도달 가능한 상태만으로 축소하는 허가가 아니다.

| 계열 | 상태 수 또는 정보 차원 | 압축 시 추가로 필요한 계약 |
| --- | --- | --- |
| DotKnot | 셀 alphabet은 `3*C+1`개지만 endpoint·경로 제약으로 유효 조합은 그보다 적음 | 색뿐 아니라 endpoint 정체성과 경로 진행 상태 보존 |
| Hanoi | `K` peg, `N` disk의 정렬된 배치는 `K**N`개 | disk별 peg 접근과 전체 rank의 decode 비용 비교 |
| LightsOut | 한 목표의 도달 가능 coset은 GF(2) move matrix rank `r`에 대해 `2**r`개 | 다른 coset 입력과 Random 목표의 문맥 처리 |
| Maze | 해당 문제에서 도달 가능한 칸 수 `R`개 | 칸 rank↔좌표와 adjacency를 InstanceContext에서 제공 |
| Pancake / TopSpin | Pancake는 `N!`, TopSpin은 생성된 부분군의 orbit 크기 | label·action 번호를 유지하는 rank/unrank |
| Rubik | sticker 배열보다 cubie·orientation·parity 제약이 강함 | 크기·색/ID·회전 metric별로 상태 공간 정의 |
| Slide | 한 목표에서 도달 가능한 순열은 `N!/2`개 (`size>=2`) | 다른 parity의 논리 입력을 유지할지 문맥에 포함 |
| Sokoban | 벽 고정, 바닥 `F`, box `B`이면 배치 상한은 `C(F,B)*(F-B)` | deadlock/도달 가능성, 벽과 player/box의 소유권 |
| TSP | 시작점 고정, `N>=2`의 도달 가능한 mask/point 쌍은 `1+(N-1)*2**(N-2)`개 | 시작 mask 한 개와 재방문 없는 partial tour의 마지막 점 구분 |
| Cayley | central state에 대한 생성기 작용의 orbit 크기 | 외부 label을 복원하는 논리 view와 orbit codec |

예를 들어 Slide의 3/4/5 크기는 한 parity의 정보 하한이 18/44/83비트로, byte 하한은
3/6/11 B다. 현재 5/8/20 B와의 차이는 순열 중복과 codec padding에서 온다. TSP 16은 위의
도달 가능 상태 수가 245761개라 18비트가 필요하며, 현재 3 B는 byte 단위 하한에 맞는다.
이는 해당 문맥의 용량 판단이지 전이 속도나 모든 raw 입력에 대한 최적성 증명이 아니다.

### DotKnot

`0`은 빈 칸, `1..2*C`는 구분되는 endpoint, `2*C+1..3*C`는 색별 경로다.
필요 폭은 `(3*C).bit_length()`이며 기본 `C=4`에서는 4비트다. endpoint의 정체성은
다음에 움직일 dot 선택에 관여하므로 색만 저장해 합치면 전이 의미가 바뀐다.
`C=1,2`에는 더 좁은 표현 여지가 있지만, 작은 비트폭의 codec padding과 전이 decode 비용을
측정해야 한다. 현재 고정 4비트가 `C>5`의 label 범위를 표현한다고 가정하지 않는다.
비가역 퍼즐이며 임의 상태 수에 대한 전역 최소성은 증명되지 않았다.

색마다 보드 전체를 비교하는 `C × size²` 중간 배열 대신, endpoint 범위에 있는 값의 최소값을
한 번 구한다. 출발 endpoint는 경로 label로, 도착은 endpoint 또는 합쳐진 경로 label로 바꾼다.
같은 byte의 서로 다른 nibble을 동시에 갱신할 수 있으므로 clear mask와 새 값을 OR로 합친다.
좌표 계산은 signed 정수에서 수행하고 전체 보드를 다시 pack하지 않는다.

### Hanoi

각 disk는 크기 순서가 고정되어 있으므로 peg 번호만 저장하면 count·빈 칸·정렬 순서를 복원할
수 있다. 정규 필드는 `disk_pegs`이며 논리 값은 `disk_pegs_unpacked`다. 생성은
`State.from_unpacked(disk_pegs=...)`로 한다. 기본 세 peg에는 disk당 2비트가 필요하며,
`(num_pegs-1).bit_length()`를 사용해 peg 수에 맞춘다. 이전 행렬의 기본 18 B를 2 B로 줄인다.
원래 행렬 자체를 3비트로 pack하는 9 B 후보보다 중복 정보를 먼저 제거하는 편이 효과적이다.

`.pegs`는 `(num_pegs, num_disks+1)`의 읽기 전용 논리 뷰다. 각 행은 count와 오름차순 disk
ID를 담으므로 기존 논리 조회·렌더링을 유지한다. 이 뷰는 전이의 저장 형식이 아니며 hot path에서
계산하지 않는다. host에 가져온 State의 뷰는 NumPy로 계산해 장치에 다시 전송하지 않는다.
과거 `State(pegs=...)` 생성과 matrix PyTree 저장 형식은 정규 입력 경로가 아니다.
논리 `pegs` 뷰의 disk ID와 count는 기존 uint8 계약이다. 압축 필드가 더 많은 disk를 담을 수
있다는 이유만으로 255개를 넘는 disk의 뷰·렌더링까지 지원한다고 가정하지 않는다.

top disk는 해당 peg에 있는 최소 disk index다. 빈 peg는 `num_disks` sentinel을 사용하면
`source_top < target_top`만으로 빈 source와 크기 규칙을 함께 판정할 수 있다. 단일 action은
두 peg만 비교하고, 전체 이웃은 모든 peg의 top을 한 번 구한다. 변경할 disk를 `where`로 선택해
기존 codec으로 pack하면 scatter를 줄이고 2/3비트 등 다른 peg 폭도 동일하게 처리할 수 있다.
전체 이웃의 codec 호출은 action 축으로 vmap한다. scalar `from_unpacked`에 action 축을
intrinsic shape의 일부인 것처럼 넣지 않는다.

디스크 0개인 기존 경계도 유지한다. 저장 배열은 0 B이고 모든 이동은 무한 비용의 self-loop이며
이미 목표 상태다. top reduction에 `num_disks` 초기값을 명시한다. 공통 field codec도 빈 intrinsic
shape를 다룰 때 `-1`로 batch 크기를 추론하지 않고 알려진 batch shape의 원소 수를 사용해야 한다.
0개 원소에서 0으로 나누는 reshape를 피하고 host/device의 빈 배열 복원을 일치시킨다.

전역 정보 하한은 `ceil(size*log2(3))`비트다. 기본 5개는 8비트이므로 현재의 2 B도 전역 최소는
아니다. base-3 rank는 더 조밀할 수 있지만, 나눗셈을 포함한 decode와 전이 비용을 별도로
측정해야 한다. 현재 표현은 빠른 disk별 접근을 유지하는 고정폭 압축이다.

### LightsOut와 Random

토글은 XOR이므로 상태를 풀지 않고 packed 바이트에 적용할 수 있다. 기존 `_move_matrix`를
little-bit order로 pack하면 경계의 이웃을 제외한 중앙·상하좌우 mask와 0 padding이 정해진다.
단일 action은 이 상수 표의 한 행만 조회하고, 전체 이웃은 표 전체와 상태를 broadcast XOR한다.
실행 시 다섯 bit 위치를 계산하는 후보도 전체 unpack보다 빨랐지만, 상수 mask 조회가 큰 배치의
산술·임시 버퍼를 더 줄였다. 상수 표는 보드가 커질 때 `O(size⁴)` 공간과 컴파일 상수 비용이
있으므로 지원 크기를 크게 확장할 때 다시 측정한다. 한 경로의 개선만으로 다른 경로의 회귀를
허용하지 않는다.
Random은 목표와 초기 표본의 차이이며 전이와 State 표현은 동일하다.

목표에서 도달 가능한 상태만 보면 GF(2) move matrix의 rank가 정보 차원을 결정한다.
rank가 셀 수보다 작으면 basis 좌표로 더 줄일 여지가 있다. 다만 일반 논리 입력에는 다른
coset의 보드도 있을 수 있고 Random 목표도 있으므로, 도달 가능성의 가정과 decode에 필요한
목표 문맥을 명시해야 한다. 1비트/셀의 고정폭 하한을 이 전역 압축의 최적성 증명으로 해석하지 않는다.

### Maze와 Room

좌표 최댓값은 `size-1`이다. 따라서 크기 256까지 uint8이고 257부터 uint16이다.
기본 Maze 23의 두 좌표를 5비트 `packed_tensor`로 바꾸면 5 B가 되어 오히려 커진다.
벽은 SolveConfig의 1비트 bitmap에 공유하고, 전이는 목적지의 한 비트만 조회한다.
음수 방향 이동은 int32에서 범위를 검사한다. uint8로 먼저 변환하면 `-1→255`가 되어
크기 256에서 잘못된 wraparound 이동이 생긴다. Room은 Maze 표현과 전이를 상속하고
요청 크기를 방 배치에 맞춰 조정하므로 `room.size`를 기준으로 용량을 계산한다.
uint16보다 큰 좌표 범위는 기존 표현의 지원 범위로 간주하지 않는다.

아주 작은 grid는 flat index나 두 nibble로 1 B에 들어갈 수도 있다. 이는 두 좌표를 그대로
저장하는 현재 byte 표현보다 작은 후보이며, 좌표 view·생성자·호출부와 decode 비용을 함께
변경해야 한다. 기본 크기에서의 2 B 결과를 모든 크기의 정보 이론적 최소라고 부르지 않는다.

좌표와 경계 검사는 signed int32로 계산한다. `size² > 2³¹`이면 flat index만 uint32로 넓혀
uint16 좌표가 가리키는 벽 주소의 overflow를 피한다. 예를 들어 46341×46341의 마지막 셀은
2147488280이며 signed int32로 표현되지 않는다. packed byte의 shift operand와 mask는
uint8로 유지한다. 인덱스 dtype을 비트 데이터에 전파하면 불필요한 정수 폭 확장이 생길 수 있다.
두 좌표의 경계는 row/column 조건으로 직접 표현해 작은 reduction과 중간 좌표 배열을 줄인다.

### PancakeSorting와 TopSpin

두 퍼즐은 `1..N` 순열이다. `N`이 2의 거듭제곱이면 0 기반 순열보다 한 비트가 더 필요하다.
기존 uint8 label의 범위는 `N<=255`다. 더 큰 순열은 생성·목표·논리 view의 별도 표현 변경이 필요하다.
Pancake 35는 6비트 field packing으로 27 B, TopSpin 20은 5비트 field packing으로 15 B
(aggregate 13 B)가 가능하다. 전체 `N!` 순열을 표현하는 permutation rank는
`ceil(log2(N!))`비트를 사용하지만 매 전이와 heuristic에서 rank/unrank 비용을 지불해야 한다.
특정 generator가 만드는 부분군의 정보 하한은 도달 가능한 상태 수로 별도 계산한다.
Pancake prefix 반전은 `source[i] = action+1-i if i<=action+1 else i`인 단일 gather다.
원소마다 scatter를 수행하는 순차 scan은 불필요하다. TopSpin의 회전/반전도 직접 인덱싱을
우선 검토한다. `.stack`/`.permutation`을 쓰는 heuristic과 생성자 계약을 유지한 채
추가 packing을 도입할 수 있다고 가정하지 않는다.
특히 Pancake의 기존 flip 명세는 `State(stack=논리 uint8 배열)` 생성과 `.stack` 조회를 직접
사용한다. packed 저장 필드로 바꾸려면 이 생성 계약과 논리 view를 함께 결정해야 한다.
현재는 byte 표현을 유지하면서 scan을 gather로 바꾼다. 추가 packing을 느리다고 기각한 것은
아니며, 생성 계약의 마이그레이션과 GPU 비교가 필요한 미채택 후보로 남긴다.

TopSpin의 일반 단일 action은 선택된 source index를 계산한다. 배치에서 `vmap(lax.switch)`가
여러 후보 배열을 만드는 비용을 줄인다. 전체 이웃에는 세 동작을 고정한 `vmap(lax.switch)`를
별도로 유지한다. 반전 구간이 0/1개 원소라 항등 변환이면 단일 action도 회전/항등 분기를
직접 사용한다. 이 경계에서는 일반 gather보다 기존 분기가 빨랐다. 반전 구간 길이는 Python
slice와 같게 정규화해 기존 음수 `turnstile_size`의 의미도 유지한다.
전체 index table의 gather는 회귀했고, roll/부분 반전을 직접 stack하는 후보는 일부 forward
배치에서 빨랐지만 작은 inverse 배치가 회귀했다. 전체 이웃의 기존 성능을 유지하면서 단일
action만 개선하는 선택도 유효하다. 단일 action의 좋은 구현을 bulk의 좋은 구현으로 일반화하지 않는다.

TopSpin 20/32를 5/6비트로 pack하는 후보는 저장량이 각각 15/24 B로 줄었지만,
decode·gather·repack이 단일 action과 전체 이웃 모두를 느리게 만들어 채택하지 않았다.
작은 permutation에서는 byte 저장이 더 나은 용량·속도 절충일 수 있다.

### RubiksCube와 Random

색 모드는 값 `0..5`에 3비트가 필요하다. tile ID 모드는 `0..6*size²-1`이다.
ID에 최소 비트폭을 적용하면 크기 2/3/4의 payload는 각각 15/42/84 B가 되지만,
비트 경계를 넘는 gather와 repack을 모든 회전마다 수행한다. byte gather 대비 GPU 속도를
확인한 뒤 채택해야 하며, 저장량만으로 tile ID의 8비트 표현을 교체하지 않는다.
색 수 보존, 고정 센터, cubie orientation/parity를 이용한 전역 압축은 현 sticker 모델과
다른 codec이다. 일반 크기, QTM/UQTM, Random 목표, Santa의 tile ID 입력을 함께 보존해야 한다.
ID 모드는 256개를 넘는 tile을 기존 uint8 표현에 넣을 수 없다.

전체 이웃에서 unpack은 공유하고 permutation 행별 gather와 기존 codec의 pack을 vmap한다.
action별 회전을 Python에서 펼친 뒤 stack하는 후보는 큰 tile-ID 모드의 forward 이웃을
빠르게 만들었지만, 작은 크기·색상 모드·inverse 이웃에서 회귀가 있어 채택하지 않았다.
public의 invalid-state 선택 방식도 XLA transpose fusion에 영향을 주므로 bulk 내부만 비교하지 않는다.

### SlidePuzzle, Random, Hard

타일은 빈 칸 0을 포함한 `0..size²-1` 순열이다. 크기 4의 nibble 표현은 8 B로 빽빽하다.
크기 5의 5비트 표현은 블록 padding 때문에 20 B이고 이상적 연속 비트열은 16 B다.
이 padding을 없애는 것은 퍼즐보다 공통 codec의 레이아웃 변경에 가깝다. 추가 decode와
단일 swap 비용, search hash/equality까지 측정해야 한다. parity로 줄어드는 상태 공간은
원소별 random access를 유지하는 표현에 자동 반영되지 않는다. Hard의 bundled 입력은
지원되는 3/4 크기와 원본 tile 순서를 유지한다.

이동은 두 tile의 swap이므로 두 값의 XOR 차이를 양쪽 비트 구간에 적용하면 된다. bit 위치는
`cell * b`이고 시작 offset의 최댓값은 `8-gcd(b,8)`이다. 따라서 필요한 byte 조각 수는
`ceil((b+8-gcd(b,8))/8)`이다. 2/4비트는 한 조각, 9/10비트는 두 조각, 11비트는 세 조각이다.
low 조각은 왼쪽 shift, 이후 조각은 `8*k-offset`만큼 오른쪽 shift해 얻는다. 각 조각을
uint8로 변환한 뒤 저장 배열에 XOR해야 넓은 논리 dtype이 raw `board`에 전파되지 않는다.
전체 이웃은 blank 탐색과 decode를 공유하고 네 방향의 mask를 독립적으로 만든다.

2×2의 단일 action은 한 바이트에서 직접 처리한다. 네 2비트 구간의 zero 여부를 scalar select로
읽어 첫 blank를 찾고, 두 구간의 XOR 차이를 교환한다. 일반 배열 decode와 argmax reduction이
필요 없어 작은 배치에서의 임시 버퍼 증가도 제거한다. 여러 zero 또는 zero가 없는 byte에서도
기존 argmax의 첫 zero/기본 index 0 선택을 유지한다. 256개 byte 패턴을 모두 비교할 수 있다.

단일 action의 6비트는 두 값을 `where`로 선택한 뒤 pack한다. 24비트 codec block에서 잘
결합되며, 6×6의 큰 배치에서 packed XOR만 사용했을 때의 회귀를 없앴다. 8비트 단일 action은
두 위치의 source index를 교환한 gather를 사용해 임시 버퍼와 작은 배치 시간을 줄인다.
5/7비트의 40/56비트 block과 넓은 값에서는 select/pack 후보가 느려 packed XOR를 유지한다.

8비트 bulk는 별도의 select/pack 경로다. packed XOR를 사용하면 13×13·15×15의 큰 배치에서
회귀가 있었으나 이 경로에서는 다른 8비트 크기와 inverse도 비회귀였다. 단일 action의 좋은
구현을 bulk에 강제하거나, 다시 pack하는 것이 항상 느리다고 가정하지 않는다.

일반 생성 입력은 크기 2~16에서, `State.from_unpacked(board=...)`로 제공하는 넓은 논리 입력은
9~16비트에서 확인했다. 넓은 값에 두 byte 조각만 적용하거나 저장 dtype을 그대로 넓히면
입력을 손상시키거나 public 조건 분기의 타입을 깨뜨린다. 기본 랜덤 생성기의 uint8 범위와
사용자가 제공한 넓은 논리 입력의 전이 지원 범위는 구분한다.

### Sokoban와 Hard

기본 10×10의 2비트 board는 25 B다. 벽과 이동 객체를 분리하고 player 위치와 box 집합만
저장하면 줄일 수 있으나, 현재 board의 정보 소유권을 InstanceContext로 옮기는 변경이다.
목표, 벽, box/player encoding과 renderer·heuristic·역전이 계약을 먼저 확인한다.
기본/Hard 데이터는 서로 다른 입력 집합이며 이론적인 동일 상태 수로 취급하지 않는다.
지원하지 않는 큰 `size`로 벤치마크를 만들어 최적성 근거로 사용하지 않는다.

전진에서는 player 탐색을 한 번 수행하고, 목적지가 빈 칸인지 또는 밀 수 있는 box인지 벡터로
판단한다. 움직이는 최대 세 칸의 2비트 필드만 clear/set하므로 보드 전체 scatter/repack과
중첩된 조건 분기가 필요 없다. box를 밀지 않는 방향에서 뒤쪽 칸을 갱신하지 않는다.

역전이도 네 pull/simple move를 독립적으로 벡터화한다. 기존 scrambling 계약은 막힌 방향이
비용 1의 self-loop이고 `filled=False`는 비용만 무한대로 만든다. 전진 public 계약과 이 차이를
최적화 과정에서 임의로 통일하지 않는다. 앞쪽 box를 pull할 때만 해당 칸을 비우고 현재 player
칸에 box를 넣으며, 이전 player 칸이 빈 칸이고 범위 안일 때만 상태를 이동한다.

### TSP

mask는 little-bit order이며 마지막 유효 byte의 완성값은 `(1 << (size % 8 or 8)) - 1`이다.
그 앞의 byte는 모두 255여야 한다. `is_solved`와 마지막 방문 판정이 이 규칙을 공유한다.
재방문은 무한 비용이며 상태 복구는 core가 담당한다. 마지막 방문 때만 시작점까지의
closing leg를 더한다. 현재 계약은 일반 edge에 `distance_matrix`, closing leg에 좌표의
Euclidean norm을 사용하므로 이 차이를 packing 변경 중에 임의로 바꾸지 않는다.
현재 point는 이미 방문한 점이라는 중복이 있지만 이를 mask에서 복원할 수는 없다.
거리 행렬의 `O(size²)` 메모리와 상태 저장량의 `O(size)`를 분리해 보고한다.

단일 action의 mask 갱신도 scatter 대신 byte index 비교로 만든 mask와 OR한다. 변경할 byte가
하나여도 immutable scatter의 복사·갱신 경로가 별도 비용을 만들 수 있다. 특히 7/8/9개 점에서
scatter 후보는 느려졌지만 OR 경로는 용량 감소와 GPU 비회귀를 함께 만족했다.

전체 이웃은 목적지마다 새 mask 전체의 완성을 검사하지 않는다. 현재 방문 수가 `size-1`이면
다음 유효 방문이 마지막이라는 사실을 한 번 계산한다. 모든 edge는 현재 point의 거리 행렬 행을
공유하고, 마지막 방문일 때만 목적지별 closing leg를 더한다. 재방문 비용은 계속 무한대다.

### CayleyPuzzle와 다섯 파생 클래스

PERMUTATION generator는 위치만 재배열하므로 central state의 값 범위가 보존된다.
비음수 최댓값이 255 이하면 uint8, 65535 이하면 uint16, 음수 또는 더 큰 값이면 int32다.
길이가 8이어도 label이 음수이거나 100000부터 시작할 수 있다. label을 `0..n-1`로
재번호화하거나 단순히 `log2(n)`비트로 줄이지 않는다. 기존 외부 논리 입력을 유지한다.

MATRIX는 central state 범위와 `modulo-1`이 모두 `0..255`에 들어갈 때 uint8을 사용하고,
그 밖에는 int32를 유지한다. uint16 저장 후보는 단일 전이에서 변환 비용에 따른 GPU 회귀가
있어 채택하지 않았다. 곱셈은 기존 int32/int64 경로와 `m*(modulo-1)^2` overflow 사전조건을 유지하고,
modulo 적용 뒤 저장 dtype으로 좁힌다. 작은 저장 dtype으로 행렬 곱을 수행하면 값이 달라진다.
inverse closure, inverse action map, `ensure_inverse_closed=False`의 비용/경고 계약도 유지한다.

CayleyPancake7/8, CayleyLRX8, CayleyTopSpin8K4, CayleyCoxeter8은 같은 adapter를 사용한다.
기본 순열은 각각 7/8 B로 저장 가능하며 파생 클래스마다 codec을 복제하지 않는다.
외부 cayleypy factory와 수작업 graph 양쪽을 확인하고, MATRIX는 별도로 확인한다.

문자열 출력은 `jax.device_get`으로 배열을 한 번 host에 가져온 후 원소를 순회한다.
GPU scalar마다 Python `int`로 변환하면 원소 수만큼 암묵적 장치 동기화가 발생한다.

## 최적화 채택과 기록

용량 또는 GPU 속도가 개선되고 다른 지표가 비회귀일 때 채택한다. 같은 GPU·seed·입력·배치에서
컴파일을 제외한 반복 중앙값을 비교한다. ±3%는 동률이며 범위를 벗어나면 재측정한다.
기본 크기, 큰 대표 크기, 비트폭·byte padding 경계를 포함한다. 단일 전이와 이웃 생성,
대표 JAxtar 탐색을 함께 본다. CPU 시간은 참고값이다.

짧은 커널은 Python dispatch와 GPU clock 변화가 wall time을 지배할 수 있다. profiler의
GPU kernel 시간과 호출 overhead를 구분하고, 같은 프로세스에서 원본/후보 순서를 번갈아
측정한다. 서로 비용이 다른 연산을 짧게 섞으면 클럭 변화가 한쪽 표본에 편중될 수 있으므로
연산별 예열 후 교차 측정한다. CPU launch와 GPU kernel은 correlation ID로 연결하며 두 trace
시계의 timestamp가 같다고 가정하지 않는다. kernel 시간만으로 전체 탐색 속도 개선을 주장하지 않는다.
trace 종료 직전의 GPU activity가 누락되는 경우도 있으므로 측정 뒤에 별도의 수집 마무리 구간을
두고, 측정 안의 CPU launch가 모두 대응하는 device 이벤트를 갖는지 확인한다. 누락된 호출을
0시간으로 계산하거나 빠진 표본을 제외한 중앙값으로 채택 여부를 판정하지 않는다.
빈 packed 필드의 0바이트 복사 API는 device 작업을 만들지 않을 수 있다. trace의 명시적인
`size:0` 기록으로 이 무연산을 구분하고 실제 kernel·비어 있지 않은 복사의 누락은 계속 검사한다.

짧은 호출만으로 GPU가 절전 상태를 벗어나지 않으면, 각 원본/후보 호출 묶음 직전에 같은 GPU
연산을 실행해 예열하고 그 시간은 측정에서 제외한다. 전역 클럭 설정을 바꾸지 않아도 이 방식으로
전력 상태 변화의 영향을 줄일 수 있다. 계산식이 같은 경로의 큰 속도 차이는 성능 이득으로
해석하기 전에 계측 조건을 확인한다.

기존 테스트는 읽고 실행할 수 있지만 작성·수정·삭제하지 않는다. 추가 경계 사례는
로컬 재현 코드로 조사하고 필요한 정식 명세는 사용자에게 제안한다. 검증 CLI·trace·원본
비교 checkout·diff·측정 보고서는 Git에서 제외된 워크스페이스 `runs/`에만 둔다.
이 문서에는 유지보수에 필요한 표현 계약과 인사이트를 남기고 실험 산출물을 편입하지 않는다.
