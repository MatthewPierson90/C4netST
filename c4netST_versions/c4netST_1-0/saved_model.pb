??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02unknown8??
?
ssquare_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namessquare_conv1/kernel
?
(ssquare_conv1/kernel/Read/ReadVariableOpReadVariableOpssquare_conv1/kernel*&
_output_shapes
:*
dtype0
|
ssquare_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namessquare_conv1/bias
u
&ssquare_conv1/bias/Read/ReadVariableOpReadVariableOpssquare_conv1/bias*
_output_shapes
:*
dtype0
?
rsquare_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namersquare_conv1/kernel
?
(rsquare_conv1/kernel/Read/ReadVariableOpReadVariableOprsquare_conv1/kernel*&
_output_shapes
:*
dtype0
|
rsquare_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namersquare_conv1/bias
u
&rsquare_conv1/bias/Read/ReadVariableOpReadVariableOprsquare_conv1/bias*
_output_shapes
:*
dtype0
?
ssquare_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namessquare_conv2/kernel
?
(ssquare_conv2/kernel/Read/ReadVariableOpReadVariableOpssquare_conv2/kernel*&
_output_shapes
:*
dtype0
|
ssquare_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namessquare_conv2/bias
u
&ssquare_conv2/bias/Read/ReadVariableOpReadVariableOpssquare_conv2/bias*
_output_shapes
:*
dtype0
?
srow_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namesrow_conv2/kernel

%srow_conv2/kernel/Read/ReadVariableOpReadVariableOpsrow_conv2/kernel*&
_output_shapes
:
*
dtype0
v
srow_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namesrow_conv2/bias
o
#srow_conv2/bias/Read/ReadVariableOpReadVariableOpsrow_conv2/bias*
_output_shapes
:
*
dtype0
?
scol_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namescol_conv2/kernel

%scol_conv2/kernel/Read/ReadVariableOpReadVariableOpscol_conv2/kernel*&
_output_shapes
:
*
dtype0
v
scol_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namescol_conv2/bias
o
#scol_conv2/bias/Read/ReadVariableOpReadVariableOpscol_conv2/bias*
_output_shapes
:
*
dtype0
?
srow_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namesrow_conv/kernel
}
$srow_conv/kernel/Read/ReadVariableOpReadVariableOpsrow_conv/kernel*&
_output_shapes
:
*
dtype0
t
srow_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namesrow_conv/bias
m
"srow_conv/bias/Read/ReadVariableOpReadVariableOpsrow_conv/bias*
_output_shapes
:
*
dtype0
?
scol_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namescol_conv/kernel
}
$scol_conv/kernel/Read/ReadVariableOpReadVariableOpscol_conv/kernel*&
_output_shapes
:
*
dtype0
t
scol_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namescol_conv/bias
m
"scol_conv/bias/Read/ReadVariableOpReadVariableOpscol_conv/bias*
_output_shapes
:
*
dtype0
?
rsquare_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namersquare_conv2/kernel
?
(rsquare_conv2/kernel/Read/ReadVariableOpReadVariableOprsquare_conv2/kernel*&
_output_shapes
:*
dtype0
|
rsquare_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namersquare_conv2/bias
u
&rsquare_conv2/bias/Read/ReadVariableOpReadVariableOprsquare_conv2/bias*
_output_shapes
:*
dtype0
?
rrow_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namerrow_conv2/kernel

%rrow_conv2/kernel/Read/ReadVariableOpReadVariableOprrow_conv2/kernel*&
_output_shapes
:
*
dtype0
v
rrow_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namerrow_conv2/bias
o
#rrow_conv2/bias/Read/ReadVariableOpReadVariableOprrow_conv2/bias*
_output_shapes
:
*
dtype0
?
rcol_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namercol_conv2/kernel

%rcol_conv2/kernel/Read/ReadVariableOpReadVariableOprcol_conv2/kernel*&
_output_shapes
:
*
dtype0
v
rcol_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namercol_conv2/bias
o
#rcol_conv2/bias/Read/ReadVariableOpReadVariableOprcol_conv2/bias*
_output_shapes
:
*
dtype0
?
rrow_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namerrow_conv/kernel
}
$rrow_conv/kernel/Read/ReadVariableOpReadVariableOprrow_conv/kernel*&
_output_shapes
:
*
dtype0
t
rrow_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namerrow_conv/bias
m
"rrow_conv/bias/Read/ReadVariableOpReadVariableOprrow_conv/bias*
_output_shapes
:
*
dtype0
?
rcol_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namercol_conv/kernel
}
$rcol_conv/kernel/Read/ReadVariableOpReadVariableOprcol_conv/kernel*&
_output_shapes
:
*
dtype0
t
rcol_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namercol_conv/bias
m
"rcol_conv/bias/Read/ReadVariableOpReadVariableOprcol_conv/bias*
_output_shapes
:
*
dtype0
?
sdense_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*$
shared_namesdense_layer/kernel
|
'sdense_layer/kernel/Read/ReadVariableOpReadVariableOpsdense_layer/kernel*
_output_shapes
:	?d*
dtype0
z
sdense_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_namesdense_layer/bias
s
%sdense_layer/bias/Read/ReadVariableOpReadVariableOpsdense_layer/bias*
_output_shapes
:d*
dtype0
?
rdense_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*$
shared_namerdense_layer/kernel
|
'rdense_layer/kernel/Read/ReadVariableOpReadVariableOprdense_layer/kernel*
_output_shapes
:	?d*
dtype0
z
rdense_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_namerdense_layer/bias
s
%rdense_layer/bias/Read/ReadVariableOpReadVariableOprdense_layer/bias*
_output_shapes
:d*
dtype0
?
score_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*$
shared_namescore_output/kernel
{
'score_output/kernel/Read/ReadVariableOpReadVariableOpscore_output/kernel*
_output_shapes

:d*
dtype0
z
score_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namescore_output/bias
s
%score_output/bias/Read/ReadVariableOpReadVariableOpscore_output/bias*
_output_shapes
:*
dtype0
?
result_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*%
shared_nameresult_output/kernel
}
(result_output/kernel/Read/ReadVariableOpReadVariableOpresult_output/kernel*
_output_shapes

:d*
dtype0
|
result_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameresult_output/bias
u
&result_output/bias/Read/ReadVariableOpReadVariableOpresult_output/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
valueރBڃ B҃
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer_with_weights-12
layer-25
layer_with_weights-13
layer-26
layer_with_weights-14
layer-27
layer_with_weights-15
layer-28
	optimizer
loss
 
signatures
#!_self_saveable_object_factories
"trainable_variables
#regularization_losses
$	variables
%	keras_api
%
#&_self_saveable_object_factories
?

'kernel
(bias
#)_self_saveable_object_factories
*trainable_variables
+regularization_losses
,	variables
-	keras_api
?

.kernel
/bias
#0_self_saveable_object_factories
1trainable_variables
2regularization_losses
3	variables
4	keras_api
?

5kernel
6bias
#7_self_saveable_object_factories
8trainable_variables
9regularization_losses
:	variables
;	keras_api
?

<kernel
=bias
#>_self_saveable_object_factories
?trainable_variables
@regularization_losses
A	variables
B	keras_api
?

Ckernel
Dbias
#E_self_saveable_object_factories
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
?

Jkernel
Kbias
#L_self_saveable_object_factories
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
?

Qkernel
Rbias
#S_self_saveable_object_factories
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
?

Xkernel
Ybias
#Z_self_saveable_object_factories
[trainable_variables
\regularization_losses
]	variables
^	keras_api
?

_kernel
`bias
#a_self_saveable_object_factories
btrainable_variables
cregularization_losses
d	variables
e	keras_api
?

fkernel
gbias
#h_self_saveable_object_factories
itrainable_variables
jregularization_losses
k	variables
l	keras_api
?

mkernel
nbias
#o_self_saveable_object_factories
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
?

tkernel
ubias
#v_self_saveable_object_factories
wtrainable_variables
xregularization_losses
y	variables
z	keras_api
w
#{_self_saveable_object_factories
|trainable_variables
}regularization_losses
~	variables
	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
 
 
 
 
?
'0
(1
.2
/3
54
65
<6
=7
C8
D9
J10
K11
Q12
R13
X14
Y15
_16
`17
f18
g19
m20
n21
t22
u23
?24
?25
?26
?27
?28
?29
?30
?31
 
?
'0
(1
.2
/3
54
65
<6
=7
C8
D9
J10
K11
Q12
R13
X14
Y15
_16
`17
f18
g19
m20
n21
t22
u23
?24
?25
?26
?27
?28
?29
?30
?31
?
 ?layer_regularization_losses
"trainable_variables
?layers
?non_trainable_variables
#regularization_losses
?layer_metrics
?metrics
$	variables
 
`^
VARIABLE_VALUEssquare_conv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEssquare_conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1
 

'0
(1
?
 ?layer_regularization_losses
*trainable_variables
?non_trainable_variables
+regularization_losses
?layer_metrics
?metrics
,	variables
?layers
`^
VARIABLE_VALUErsquare_conv1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUErsquare_conv1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1
 

.0
/1
?
 ?layer_regularization_losses
1trainable_variables
?non_trainable_variables
2regularization_losses
?layer_metrics
?metrics
3	variables
?layers
`^
VARIABLE_VALUEssquare_conv2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEssquare_conv2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61
 

50
61
?
 ?layer_regularization_losses
8trainable_variables
?non_trainable_variables
9regularization_losses
?layer_metrics
?metrics
:	variables
?layers
][
VARIABLE_VALUEsrow_conv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsrow_conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1
 

<0
=1
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
@regularization_losses
?layer_metrics
?metrics
A	variables
?layers
][
VARIABLE_VALUEscol_conv2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEscol_conv2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1
 

C0
D1
?
 ?layer_regularization_losses
Ftrainable_variables
?non_trainable_variables
Gregularization_losses
?layer_metrics
?metrics
H	variables
?layers
\Z
VARIABLE_VALUEsrow_conv/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEsrow_conv/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1
 

J0
K1
?
 ?layer_regularization_losses
Mtrainable_variables
?non_trainable_variables
Nregularization_losses
?layer_metrics
?metrics
O	variables
?layers
\Z
VARIABLE_VALUEscol_conv/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEscol_conv/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1
 

Q0
R1
?
 ?layer_regularization_losses
Ttrainable_variables
?non_trainable_variables
Uregularization_losses
?layer_metrics
?metrics
V	variables
?layers
`^
VARIABLE_VALUErsquare_conv2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUErsquare_conv2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1
 

X0
Y1
?
 ?layer_regularization_losses
[trainable_variables
?non_trainable_variables
\regularization_losses
?layer_metrics
?metrics
]	variables
?layers
][
VARIABLE_VALUErrow_conv2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUErrow_conv2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

_0
`1
 

_0
`1
?
 ?layer_regularization_losses
btrainable_variables
?non_trainable_variables
cregularization_losses
?layer_metrics
?metrics
d	variables
?layers
][
VARIABLE_VALUErcol_conv2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUErcol_conv2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

f0
g1
 

f0
g1
?
 ?layer_regularization_losses
itrainable_variables
?non_trainable_variables
jregularization_losses
?layer_metrics
?metrics
k	variables
?layers
][
VARIABLE_VALUErrow_conv/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUErrow_conv/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1
 

m0
n1
?
 ?layer_regularization_losses
ptrainable_variables
?non_trainable_variables
qregularization_losses
?layer_metrics
?metrics
r	variables
?layers
][
VARIABLE_VALUErcol_conv/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUErcol_conv/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

t0
u1
 

t0
u1
?
 ?layer_regularization_losses
wtrainable_variables
?non_trainable_variables
xregularization_losses
?layer_metrics
?metrics
y	variables
?layers
 
 
 
 
?
 ?layer_regularization_losses
|trainable_variables
?non_trainable_variables
}regularization_losses
?layer_metrics
?metrics
~	variables
?layers
 
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
 
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
 
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
 
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
 
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
 
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
 
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
 
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
 
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
 
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
 
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
`^
VARIABLE_VALUEsdense_layer/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEsdense_layer/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
`^
VARIABLE_VALUErdense_layer/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUErdense_layer/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
`^
VARIABLE_VALUEscore_output/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEscore_output/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
a_
VARIABLE_VALUEresult_output/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEresult_output/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
 
 
8
?0
?1
?2
?3
?4
?5
?6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_54keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_54keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_64keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_64keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
?
serving_default_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputrsquare_conv1/kernelrsquare_conv1/biasssquare_conv1/kernelssquare_conv1/biasrcol_conv/kernelrcol_conv/biasrrow_conv/kernelrrow_conv/biasrcol_conv2/kernelrcol_conv2/biasrrow_conv2/kernelrrow_conv2/biasrsquare_conv2/kernelrsquare_conv2/biasscol_conv/kernelscol_conv/biassrow_conv/kernelsrow_conv/biasscol_conv2/kernelscol_conv2/biassrow_conv2/kernelsrow_conv2/biasssquare_conv2/kernelssquare_conv2/biasrdense_layer/kernelrdense_layer/biassdense_layer/kernelsdense_layer/biasresult_output/kernelresult_output/biasscore_output/kernelscore_output/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_26791
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(ssquare_conv1/kernel/Read/ReadVariableOp&ssquare_conv1/bias/Read/ReadVariableOp(rsquare_conv1/kernel/Read/ReadVariableOp&rsquare_conv1/bias/Read/ReadVariableOp(ssquare_conv2/kernel/Read/ReadVariableOp&ssquare_conv2/bias/Read/ReadVariableOp%srow_conv2/kernel/Read/ReadVariableOp#srow_conv2/bias/Read/ReadVariableOp%scol_conv2/kernel/Read/ReadVariableOp#scol_conv2/bias/Read/ReadVariableOp$srow_conv/kernel/Read/ReadVariableOp"srow_conv/bias/Read/ReadVariableOp$scol_conv/kernel/Read/ReadVariableOp"scol_conv/bias/Read/ReadVariableOp(rsquare_conv2/kernel/Read/ReadVariableOp&rsquare_conv2/bias/Read/ReadVariableOp%rrow_conv2/kernel/Read/ReadVariableOp#rrow_conv2/bias/Read/ReadVariableOp%rcol_conv2/kernel/Read/ReadVariableOp#rcol_conv2/bias/Read/ReadVariableOp$rrow_conv/kernel/Read/ReadVariableOp"rrow_conv/bias/Read/ReadVariableOp$rcol_conv/kernel/Read/ReadVariableOp"rcol_conv/bias/Read/ReadVariableOp'sdense_layer/kernel/Read/ReadVariableOp%sdense_layer/bias/Read/ReadVariableOp'rdense_layer/kernel/Read/ReadVariableOp%rdense_layer/bias/Read/ReadVariableOp'score_output/kernel/Read/ReadVariableOp%score_output/bias/Read/ReadVariableOp(result_output/kernel/Read/ReadVariableOp&result_output/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_5/Read/ReadVariableOpcount_5/Read/ReadVariableOptotal_6/Read/ReadVariableOpcount_6/Read/ReadVariableOpConst*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_27845
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamessquare_conv1/kernelssquare_conv1/biasrsquare_conv1/kernelrsquare_conv1/biasssquare_conv2/kernelssquare_conv2/biassrow_conv2/kernelsrow_conv2/biasscol_conv2/kernelscol_conv2/biassrow_conv/kernelsrow_conv/biasscol_conv/kernelscol_conv/biasrsquare_conv2/kernelrsquare_conv2/biasrrow_conv2/kernelrrow_conv2/biasrcol_conv2/kernelrcol_conv2/biasrrow_conv/kernelrrow_conv/biasrcol_conv/kernelrcol_conv/biassdense_layer/kernelsdense_layer/biasrdense_layer/kernelrdense_layer/biasscore_output/kernelscore_output/biasresult_output/kernelresult_output/biastotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4total_5count_5total_6count_6*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_27993??
?
?
D__inference_rcol_conv_layer_call_and_return_conditional_losses_25572

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_rsquare_conv2_layer_call_fn_27375

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rsquare_conv2_layer_call_and_return_conditional_losses_256402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_5_layer_call_and_return_conditional_losses_25809

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????F   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????F2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
E__inference_rrow_conv2_layer_call_and_return_conditional_losses_27386

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_ssquare_conv1_layer_call_and_return_conditional_losses_25555

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_rrow_conv_layer_call_and_return_conditional_losses_25589

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_7_layer_call_and_return_conditional_losses_25737

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????(2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_sdense_layer_layer_call_and_return_conditional_losses_27614

inputs1
matmul_readvariableop_resource:	?d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_4_layer_call_fn_27499

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_258012
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
օ
?
@__inference_c4net_layer_call_and_return_conditional_losses_26621	
input-
rsquare_conv1_26527:!
rsquare_conv1_26529:-
ssquare_conv1_26532:!
ssquare_conv1_26534:)
rcol_conv_26537:

rcol_conv_26539:
)
rrow_conv_26542:

rrow_conv_26544:
*
rcol_conv2_26547:

rcol_conv2_26549:
*
rrow_conv2_26552:

rrow_conv2_26554:
-
rsquare_conv2_26557:!
rsquare_conv2_26559:)
scol_conv_26562:

scol_conv_26564:
)
srow_conv_26567:

srow_conv_26569:
*
scol_conv2_26572:

scol_conv2_26574:
*
srow_conv2_26577:

srow_conv2_26579:
-
ssquare_conv2_26582:!
ssquare_conv2_26584:%
rdense_layer_26599:	?d 
rdense_layer_26601:d%
sdense_layer_26604:	?d 
sdense_layer_26606:d%
result_output_26609:d!
result_output_26611:$
score_output_26614:d 
score_output_26616:
identity

identity_1??!rcol_conv/StatefulPartitionedCall?"rcol_conv2/StatefulPartitionedCall?$rdense_layer/StatefulPartitionedCall?%result_output/StatefulPartitionedCall?!rrow_conv/StatefulPartitionedCall?"rrow_conv2/StatefulPartitionedCall?%rsquare_conv1/StatefulPartitionedCall?%rsquare_conv2/StatefulPartitionedCall?!scol_conv/StatefulPartitionedCall?"scol_conv2/StatefulPartitionedCall?$score_output/StatefulPartitionedCall?$sdense_layer/StatefulPartitionedCall?!srow_conv/StatefulPartitionedCall?"srow_conv2/StatefulPartitionedCall?%ssquare_conv1/StatefulPartitionedCall?%ssquare_conv2/StatefulPartitionedCall?
%rsquare_conv1/StatefulPartitionedCallStatefulPartitionedCallinputrsquare_conv1_26527rsquare_conv1_26529*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rsquare_conv1_layer_call_and_return_conditional_losses_255382'
%rsquare_conv1/StatefulPartitionedCall?
%ssquare_conv1/StatefulPartitionedCallStatefulPartitionedCallinputssquare_conv1_26532ssquare_conv1_26534*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_ssquare_conv1_layer_call_and_return_conditional_losses_255552'
%ssquare_conv1/StatefulPartitionedCall?
!rcol_conv/StatefulPartitionedCallStatefulPartitionedCallinputrcol_conv_26537rcol_conv_26539*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_rcol_conv_layer_call_and_return_conditional_losses_255722#
!rcol_conv/StatefulPartitionedCall?
!rrow_conv/StatefulPartitionedCallStatefulPartitionedCallinputrrow_conv_26542rrow_conv_26544*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_rrow_conv_layer_call_and_return_conditional_losses_255892#
!rrow_conv/StatefulPartitionedCall?
"rcol_conv2/StatefulPartitionedCallStatefulPartitionedCall.rsquare_conv1/StatefulPartitionedCall:output:0rcol_conv2_26547rcol_conv2_26549*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_rcol_conv2_layer_call_and_return_conditional_losses_256062$
"rcol_conv2/StatefulPartitionedCall?
"rrow_conv2/StatefulPartitionedCallStatefulPartitionedCall.rsquare_conv1/StatefulPartitionedCall:output:0rrow_conv2_26552rrow_conv2_26554*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_rrow_conv2_layer_call_and_return_conditional_losses_256232$
"rrow_conv2/StatefulPartitionedCall?
%rsquare_conv2/StatefulPartitionedCallStatefulPartitionedCall.rsquare_conv1/StatefulPartitionedCall:output:0rsquare_conv2_26557rsquare_conv2_26559*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rsquare_conv2_layer_call_and_return_conditional_losses_256402'
%rsquare_conv2/StatefulPartitionedCall?
!scol_conv/StatefulPartitionedCallStatefulPartitionedCallinputscol_conv_26562scol_conv_26564*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_scol_conv_layer_call_and_return_conditional_losses_256572#
!scol_conv/StatefulPartitionedCall?
!srow_conv/StatefulPartitionedCallStatefulPartitionedCallinputsrow_conv_26567srow_conv_26569*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_srow_conv_layer_call_and_return_conditional_losses_256742#
!srow_conv/StatefulPartitionedCall?
"scol_conv2/StatefulPartitionedCallStatefulPartitionedCall.ssquare_conv1/StatefulPartitionedCall:output:0scol_conv2_26572scol_conv2_26574*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_scol_conv2_layer_call_and_return_conditional_losses_256912$
"scol_conv2/StatefulPartitionedCall?
"srow_conv2/StatefulPartitionedCallStatefulPartitionedCall.ssquare_conv1/StatefulPartitionedCall:output:0srow_conv2_26577srow_conv2_26579*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_srow_conv2_layer_call_and_return_conditional_losses_257082$
"srow_conv2/StatefulPartitionedCall?
%ssquare_conv2/StatefulPartitionedCallStatefulPartitionedCall.ssquare_conv1/StatefulPartitionedCall:output:0ssquare_conv2_26582ssquare_conv2_26584*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_ssquare_conv2_layer_call_and_return_conditional_losses_257252'
%ssquare_conv2/StatefulPartitionedCall?
flatten_7/PartitionedCallPartitionedCall.rsquare_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_257372
flatten_7/PartitionedCall?
flatten_8/PartitionedCallPartitionedCall+rrow_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_257452
flatten_8/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall+rcol_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_9_layer_call_and_return_conditional_losses_257532
flatten_9/PartitionedCall?
flatten_10/PartitionedCallPartitionedCall*rrow_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_10_layer_call_and_return_conditional_losses_257612
flatten_10/PartitionedCall?
flatten_11/PartitionedCallPartitionedCall*rcol_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_11_layer_call_and_return_conditional_losses_257692
flatten_11/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall.ssquare_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_257772
flatten_1/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall+srow_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_257852
flatten_2/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall+scol_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_257932
flatten_3/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall*srow_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_258012
flatten_4/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall*scol_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_258092
flatten_5/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall"flatten_7/PartitionedCall:output:0"flatten_8/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0#flatten_10/PartitionedCall:output:0#flatten_11/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_258212
concatenate_1/PartitionedCall?
concatenate/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_258332
concatenate/PartitionedCall?
$rdense_layer/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0rdense_layer_26599rdense_layer_26601*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_rdense_layer_layer_call_and_return_conditional_losses_258462&
$rdense_layer/StatefulPartitionedCall?
$sdense_layer/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sdense_layer_26604sdense_layer_26606*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sdense_layer_layer_call_and_return_conditional_losses_258632&
$sdense_layer/StatefulPartitionedCall?
%result_output/StatefulPartitionedCallStatefulPartitionedCall-rdense_layer/StatefulPartitionedCall:output:0result_output_26609result_output_26611*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_result_output_layer_call_and_return_conditional_losses_258802'
%result_output/StatefulPartitionedCall?
$score_output/StatefulPartitionedCallStatefulPartitionedCall-sdense_layer/StatefulPartitionedCall:output:0score_output_26614score_output_26616*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_score_output_layer_call_and_return_conditional_losses_258972&
$score_output/StatefulPartitionedCall?
IdentityIdentity-score_output/StatefulPartitionedCall:output:0"^rcol_conv/StatefulPartitionedCall#^rcol_conv2/StatefulPartitionedCall%^rdense_layer/StatefulPartitionedCall&^result_output/StatefulPartitionedCall"^rrow_conv/StatefulPartitionedCall#^rrow_conv2/StatefulPartitionedCall&^rsquare_conv1/StatefulPartitionedCall&^rsquare_conv2/StatefulPartitionedCall"^scol_conv/StatefulPartitionedCall#^scol_conv2/StatefulPartitionedCall%^score_output/StatefulPartitionedCall%^sdense_layer/StatefulPartitionedCall"^srow_conv/StatefulPartitionedCall#^srow_conv2/StatefulPartitionedCall&^ssquare_conv1/StatefulPartitionedCall&^ssquare_conv2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity.result_output/StatefulPartitionedCall:output:0"^rcol_conv/StatefulPartitionedCall#^rcol_conv2/StatefulPartitionedCall%^rdense_layer/StatefulPartitionedCall&^result_output/StatefulPartitionedCall"^rrow_conv/StatefulPartitionedCall#^rrow_conv2/StatefulPartitionedCall&^rsquare_conv1/StatefulPartitionedCall&^rsquare_conv2/StatefulPartitionedCall"^scol_conv/StatefulPartitionedCall#^scol_conv2/StatefulPartitionedCall%^score_output/StatefulPartitionedCall%^sdense_layer/StatefulPartitionedCall"^srow_conv/StatefulPartitionedCall#^srow_conv2/StatefulPartitionedCall&^ssquare_conv1/StatefulPartitionedCall&^ssquare_conv2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!rcol_conv/StatefulPartitionedCall!rcol_conv/StatefulPartitionedCall2H
"rcol_conv2/StatefulPartitionedCall"rcol_conv2/StatefulPartitionedCall2L
$rdense_layer/StatefulPartitionedCall$rdense_layer/StatefulPartitionedCall2N
%result_output/StatefulPartitionedCall%result_output/StatefulPartitionedCall2F
!rrow_conv/StatefulPartitionedCall!rrow_conv/StatefulPartitionedCall2H
"rrow_conv2/StatefulPartitionedCall"rrow_conv2/StatefulPartitionedCall2N
%rsquare_conv1/StatefulPartitionedCall%rsquare_conv1/StatefulPartitionedCall2N
%rsquare_conv2/StatefulPartitionedCall%rsquare_conv2/StatefulPartitionedCall2F
!scol_conv/StatefulPartitionedCall!scol_conv/StatefulPartitionedCall2H
"scol_conv2/StatefulPartitionedCall"scol_conv2/StatefulPartitionedCall2L
$score_output/StatefulPartitionedCall$score_output/StatefulPartitionedCall2L
$sdense_layer/StatefulPartitionedCall$sdense_layer/StatefulPartitionedCall2F
!srow_conv/StatefulPartitionedCall!srow_conv/StatefulPartitionedCall2H
"srow_conv2/StatefulPartitionedCall"srow_conv2/StatefulPartitionedCall2N
%ssquare_conv1/StatefulPartitionedCall%ssquare_conv1/StatefulPartitionedCall2N
%ssquare_conv2/StatefulPartitionedCall%ssquare_conv2/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
ޅ
?
@__inference_c4net_layer_call_and_return_conditional_losses_25905

inputs-
rsquare_conv1_25539:!
rsquare_conv1_25541:-
ssquare_conv1_25556:!
ssquare_conv1_25558:)
rcol_conv_25573:

rcol_conv_25575:
)
rrow_conv_25590:

rrow_conv_25592:
*
rcol_conv2_25607:

rcol_conv2_25609:
*
rrow_conv2_25624:

rrow_conv2_25626:
-
rsquare_conv2_25641:!
rsquare_conv2_25643:)
scol_conv_25658:

scol_conv_25660:
)
srow_conv_25675:

srow_conv_25677:
*
scol_conv2_25692:

scol_conv2_25694:
*
srow_conv2_25709:

srow_conv2_25711:
-
ssquare_conv2_25726:!
ssquare_conv2_25728:%
rdense_layer_25847:	?d 
rdense_layer_25849:d%
sdense_layer_25864:	?d 
sdense_layer_25866:d%
result_output_25881:d!
result_output_25883:$
score_output_25898:d 
score_output_25900:
identity

identity_1??!rcol_conv/StatefulPartitionedCall?"rcol_conv2/StatefulPartitionedCall?$rdense_layer/StatefulPartitionedCall?%result_output/StatefulPartitionedCall?!rrow_conv/StatefulPartitionedCall?"rrow_conv2/StatefulPartitionedCall?%rsquare_conv1/StatefulPartitionedCall?%rsquare_conv2/StatefulPartitionedCall?!scol_conv/StatefulPartitionedCall?"scol_conv2/StatefulPartitionedCall?$score_output/StatefulPartitionedCall?$sdense_layer/StatefulPartitionedCall?!srow_conv/StatefulPartitionedCall?"srow_conv2/StatefulPartitionedCall?%ssquare_conv1/StatefulPartitionedCall?%ssquare_conv2/StatefulPartitionedCall?
%rsquare_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsrsquare_conv1_25539rsquare_conv1_25541*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rsquare_conv1_layer_call_and_return_conditional_losses_255382'
%rsquare_conv1/StatefulPartitionedCall?
%ssquare_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsssquare_conv1_25556ssquare_conv1_25558*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_ssquare_conv1_layer_call_and_return_conditional_losses_255552'
%ssquare_conv1/StatefulPartitionedCall?
!rcol_conv/StatefulPartitionedCallStatefulPartitionedCallinputsrcol_conv_25573rcol_conv_25575*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_rcol_conv_layer_call_and_return_conditional_losses_255722#
!rcol_conv/StatefulPartitionedCall?
!rrow_conv/StatefulPartitionedCallStatefulPartitionedCallinputsrrow_conv_25590rrow_conv_25592*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_rrow_conv_layer_call_and_return_conditional_losses_255892#
!rrow_conv/StatefulPartitionedCall?
"rcol_conv2/StatefulPartitionedCallStatefulPartitionedCall.rsquare_conv1/StatefulPartitionedCall:output:0rcol_conv2_25607rcol_conv2_25609*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_rcol_conv2_layer_call_and_return_conditional_losses_256062$
"rcol_conv2/StatefulPartitionedCall?
"rrow_conv2/StatefulPartitionedCallStatefulPartitionedCall.rsquare_conv1/StatefulPartitionedCall:output:0rrow_conv2_25624rrow_conv2_25626*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_rrow_conv2_layer_call_and_return_conditional_losses_256232$
"rrow_conv2/StatefulPartitionedCall?
%rsquare_conv2/StatefulPartitionedCallStatefulPartitionedCall.rsquare_conv1/StatefulPartitionedCall:output:0rsquare_conv2_25641rsquare_conv2_25643*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rsquare_conv2_layer_call_and_return_conditional_losses_256402'
%rsquare_conv2/StatefulPartitionedCall?
!scol_conv/StatefulPartitionedCallStatefulPartitionedCallinputsscol_conv_25658scol_conv_25660*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_scol_conv_layer_call_and_return_conditional_losses_256572#
!scol_conv/StatefulPartitionedCall?
!srow_conv/StatefulPartitionedCallStatefulPartitionedCallinputssrow_conv_25675srow_conv_25677*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_srow_conv_layer_call_and_return_conditional_losses_256742#
!srow_conv/StatefulPartitionedCall?
"scol_conv2/StatefulPartitionedCallStatefulPartitionedCall.ssquare_conv1/StatefulPartitionedCall:output:0scol_conv2_25692scol_conv2_25694*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_scol_conv2_layer_call_and_return_conditional_losses_256912$
"scol_conv2/StatefulPartitionedCall?
"srow_conv2/StatefulPartitionedCallStatefulPartitionedCall.ssquare_conv1/StatefulPartitionedCall:output:0srow_conv2_25709srow_conv2_25711*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_srow_conv2_layer_call_and_return_conditional_losses_257082$
"srow_conv2/StatefulPartitionedCall?
%ssquare_conv2/StatefulPartitionedCallStatefulPartitionedCall.ssquare_conv1/StatefulPartitionedCall:output:0ssquare_conv2_25726ssquare_conv2_25728*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_ssquare_conv2_layer_call_and_return_conditional_losses_257252'
%ssquare_conv2/StatefulPartitionedCall?
flatten_7/PartitionedCallPartitionedCall.rsquare_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_257372
flatten_7/PartitionedCall?
flatten_8/PartitionedCallPartitionedCall+rrow_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_257452
flatten_8/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall+rcol_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_9_layer_call_and_return_conditional_losses_257532
flatten_9/PartitionedCall?
flatten_10/PartitionedCallPartitionedCall*rrow_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_10_layer_call_and_return_conditional_losses_257612
flatten_10/PartitionedCall?
flatten_11/PartitionedCallPartitionedCall*rcol_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_11_layer_call_and_return_conditional_losses_257692
flatten_11/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall.ssquare_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_257772
flatten_1/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall+srow_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_257852
flatten_2/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall+scol_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_257932
flatten_3/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall*srow_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_258012
flatten_4/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall*scol_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_258092
flatten_5/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall"flatten_7/PartitionedCall:output:0"flatten_8/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0#flatten_10/PartitionedCall:output:0#flatten_11/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_258212
concatenate_1/PartitionedCall?
concatenate/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_258332
concatenate/PartitionedCall?
$rdense_layer/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0rdense_layer_25847rdense_layer_25849*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_rdense_layer_layer_call_and_return_conditional_losses_258462&
$rdense_layer/StatefulPartitionedCall?
$sdense_layer/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sdense_layer_25864sdense_layer_25866*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sdense_layer_layer_call_and_return_conditional_losses_258632&
$sdense_layer/StatefulPartitionedCall?
%result_output/StatefulPartitionedCallStatefulPartitionedCall-rdense_layer/StatefulPartitionedCall:output:0result_output_25881result_output_25883*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_result_output_layer_call_and_return_conditional_losses_258802'
%result_output/StatefulPartitionedCall?
$score_output/StatefulPartitionedCallStatefulPartitionedCall-sdense_layer/StatefulPartitionedCall:output:0score_output_25898score_output_25900*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_score_output_layer_call_and_return_conditional_losses_258972&
$score_output/StatefulPartitionedCall?
IdentityIdentity-score_output/StatefulPartitionedCall:output:0"^rcol_conv/StatefulPartitionedCall#^rcol_conv2/StatefulPartitionedCall%^rdense_layer/StatefulPartitionedCall&^result_output/StatefulPartitionedCall"^rrow_conv/StatefulPartitionedCall#^rrow_conv2/StatefulPartitionedCall&^rsquare_conv1/StatefulPartitionedCall&^rsquare_conv2/StatefulPartitionedCall"^scol_conv/StatefulPartitionedCall#^scol_conv2/StatefulPartitionedCall%^score_output/StatefulPartitionedCall%^sdense_layer/StatefulPartitionedCall"^srow_conv/StatefulPartitionedCall#^srow_conv2/StatefulPartitionedCall&^ssquare_conv1/StatefulPartitionedCall&^ssquare_conv2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity.result_output/StatefulPartitionedCall:output:0"^rcol_conv/StatefulPartitionedCall#^rcol_conv2/StatefulPartitionedCall%^rdense_layer/StatefulPartitionedCall&^result_output/StatefulPartitionedCall"^rrow_conv/StatefulPartitionedCall#^rrow_conv2/StatefulPartitionedCall&^rsquare_conv1/StatefulPartitionedCall&^rsquare_conv2/StatefulPartitionedCall"^scol_conv/StatefulPartitionedCall#^scol_conv2/StatefulPartitionedCall%^score_output/StatefulPartitionedCall%^sdense_layer/StatefulPartitionedCall"^srow_conv/StatefulPartitionedCall#^srow_conv2/StatefulPartitionedCall&^ssquare_conv1/StatefulPartitionedCall&^ssquare_conv2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!rcol_conv/StatefulPartitionedCall!rcol_conv/StatefulPartitionedCall2H
"rcol_conv2/StatefulPartitionedCall"rcol_conv2/StatefulPartitionedCall2L
$rdense_layer/StatefulPartitionedCall$rdense_layer/StatefulPartitionedCall2N
%result_output/StatefulPartitionedCall%result_output/StatefulPartitionedCall2F
!rrow_conv/StatefulPartitionedCall!rrow_conv/StatefulPartitionedCall2H
"rrow_conv2/StatefulPartitionedCall"rrow_conv2/StatefulPartitionedCall2N
%rsquare_conv1/StatefulPartitionedCall%rsquare_conv1/StatefulPartitionedCall2N
%rsquare_conv2/StatefulPartitionedCall%rsquare_conv2/StatefulPartitionedCall2F
!scol_conv/StatefulPartitionedCall!scol_conv/StatefulPartitionedCall2H
"scol_conv2/StatefulPartitionedCall"scol_conv2/StatefulPartitionedCall2L
$score_output/StatefulPartitionedCall$score_output/StatefulPartitionedCall2L
$sdense_layer/StatefulPartitionedCall$sdense_layer/StatefulPartitionedCall2F
!srow_conv/StatefulPartitionedCall!srow_conv/StatefulPartitionedCall2H
"srow_conv2/StatefulPartitionedCall"srow_conv2/StatefulPartitionedCall2N
%ssquare_conv1/StatefulPartitionedCall%ssquare_conv1/StatefulPartitionedCall2N
%ssquare_conv2/StatefulPartitionedCall%ssquare_conv2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_sdense_layer_layer_call_and_return_conditional_losses_25863

inputs1
matmul_readvariableop_resource:	?d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
G__inference_rdense_layer_layer_call_and_return_conditional_losses_27634

inputs1
matmul_readvariableop_resource:	?d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_rcol_conv2_layer_call_fn_27415

inputs!
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_rcol_conv2_layer_call_and_return_conditional_losses_256062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_rdense_layer_layer_call_and_return_conditional_losses_25846

inputs1
matmul_readvariableop_resource:	?d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_ssquare_conv1_layer_call_and_return_conditional_losses_27226

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_srow_conv_layer_call_and_return_conditional_losses_25674

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_rsquare_conv2_layer_call_and_return_conditional_losses_27366

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_c4net_layer_call_fn_25974	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:

	unknown_4:
#
	unknown_5:

	unknown_6:
#
	unknown_7:

	unknown_8:
#
	unknown_9:


unknown_10:
$

unknown_11:

unknown_12:$

unknown_13:


unknown_14:
$

unknown_15:


unknown_16:
$

unknown_17:


unknown_18:
$

unknown_19:


unknown_20:
$

unknown_21:

unknown_22:

unknown_23:	?d

unknown_24:d

unknown_25:	?d

unknown_26:d

unknown_27:d

unknown_28:

unknown_29:d

unknown_30:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_c4net_layer_call_and_return_conditional_losses_259052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
??
?
 __inference__wrapped_model_25520	
inputL
2c4net_rsquare_conv1_conv2d_readvariableop_resource:A
3c4net_rsquare_conv1_biasadd_readvariableop_resource:L
2c4net_ssquare_conv1_conv2d_readvariableop_resource:A
3c4net_ssquare_conv1_biasadd_readvariableop_resource:H
.c4net_rcol_conv_conv2d_readvariableop_resource:
=
/c4net_rcol_conv_biasadd_readvariableop_resource:
H
.c4net_rrow_conv_conv2d_readvariableop_resource:
=
/c4net_rrow_conv_biasadd_readvariableop_resource:
I
/c4net_rcol_conv2_conv2d_readvariableop_resource:
>
0c4net_rcol_conv2_biasadd_readvariableop_resource:
I
/c4net_rrow_conv2_conv2d_readvariableop_resource:
>
0c4net_rrow_conv2_biasadd_readvariableop_resource:
L
2c4net_rsquare_conv2_conv2d_readvariableop_resource:A
3c4net_rsquare_conv2_biasadd_readvariableop_resource:H
.c4net_scol_conv_conv2d_readvariableop_resource:
=
/c4net_scol_conv_biasadd_readvariableop_resource:
H
.c4net_srow_conv_conv2d_readvariableop_resource:
=
/c4net_srow_conv_biasadd_readvariableop_resource:
I
/c4net_scol_conv2_conv2d_readvariableop_resource:
>
0c4net_scol_conv2_biasadd_readvariableop_resource:
I
/c4net_srow_conv2_conv2d_readvariableop_resource:
>
0c4net_srow_conv2_biasadd_readvariableop_resource:
L
2c4net_ssquare_conv2_conv2d_readvariableop_resource:A
3c4net_ssquare_conv2_biasadd_readvariableop_resource:D
1c4net_rdense_layer_matmul_readvariableop_resource:	?d@
2c4net_rdense_layer_biasadd_readvariableop_resource:dD
1c4net_sdense_layer_matmul_readvariableop_resource:	?d@
2c4net_sdense_layer_biasadd_readvariableop_resource:dD
2c4net_result_output_matmul_readvariableop_resource:dA
3c4net_result_output_biasadd_readvariableop_resource:C
1c4net_score_output_matmul_readvariableop_resource:d@
2c4net_score_output_biasadd_readvariableop_resource:
identity

identity_1??&c4net/rcol_conv/BiasAdd/ReadVariableOp?%c4net/rcol_conv/Conv2D/ReadVariableOp?'c4net/rcol_conv2/BiasAdd/ReadVariableOp?&c4net/rcol_conv2/Conv2D/ReadVariableOp?)c4net/rdense_layer/BiasAdd/ReadVariableOp?(c4net/rdense_layer/MatMul/ReadVariableOp?*c4net/result_output/BiasAdd/ReadVariableOp?)c4net/result_output/MatMul/ReadVariableOp?&c4net/rrow_conv/BiasAdd/ReadVariableOp?%c4net/rrow_conv/Conv2D/ReadVariableOp?'c4net/rrow_conv2/BiasAdd/ReadVariableOp?&c4net/rrow_conv2/Conv2D/ReadVariableOp?*c4net/rsquare_conv1/BiasAdd/ReadVariableOp?)c4net/rsquare_conv1/Conv2D/ReadVariableOp?*c4net/rsquare_conv2/BiasAdd/ReadVariableOp?)c4net/rsquare_conv2/Conv2D/ReadVariableOp?&c4net/scol_conv/BiasAdd/ReadVariableOp?%c4net/scol_conv/Conv2D/ReadVariableOp?'c4net/scol_conv2/BiasAdd/ReadVariableOp?&c4net/scol_conv2/Conv2D/ReadVariableOp?)c4net/score_output/BiasAdd/ReadVariableOp?(c4net/score_output/MatMul/ReadVariableOp?)c4net/sdense_layer/BiasAdd/ReadVariableOp?(c4net/sdense_layer/MatMul/ReadVariableOp?&c4net/srow_conv/BiasAdd/ReadVariableOp?%c4net/srow_conv/Conv2D/ReadVariableOp?'c4net/srow_conv2/BiasAdd/ReadVariableOp?&c4net/srow_conv2/Conv2D/ReadVariableOp?*c4net/ssquare_conv1/BiasAdd/ReadVariableOp?)c4net/ssquare_conv1/Conv2D/ReadVariableOp?*c4net/ssquare_conv2/BiasAdd/ReadVariableOp?)c4net/ssquare_conv2/Conv2D/ReadVariableOp?
)c4net/rsquare_conv1/Conv2D/ReadVariableOpReadVariableOp2c4net_rsquare_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)c4net/rsquare_conv1/Conv2D/ReadVariableOp?
c4net/rsquare_conv1/Conv2DConv2Dinput1c4net/rsquare_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
c4net/rsquare_conv1/Conv2D?
*c4net/rsquare_conv1/BiasAdd/ReadVariableOpReadVariableOp3c4net_rsquare_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*c4net/rsquare_conv1/BiasAdd/ReadVariableOp?
c4net/rsquare_conv1/BiasAddBiasAdd#c4net/rsquare_conv1/Conv2D:output:02c4net/rsquare_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
c4net/rsquare_conv1/BiasAdd?
c4net/rsquare_conv1/ReluRelu$c4net/rsquare_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
c4net/rsquare_conv1/Relu?
)c4net/ssquare_conv1/Conv2D/ReadVariableOpReadVariableOp2c4net_ssquare_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)c4net/ssquare_conv1/Conv2D/ReadVariableOp?
c4net/ssquare_conv1/Conv2DConv2Dinput1c4net/ssquare_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
c4net/ssquare_conv1/Conv2D?
*c4net/ssquare_conv1/BiasAdd/ReadVariableOpReadVariableOp3c4net_ssquare_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*c4net/ssquare_conv1/BiasAdd/ReadVariableOp?
c4net/ssquare_conv1/BiasAddBiasAdd#c4net/ssquare_conv1/Conv2D:output:02c4net/ssquare_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
c4net/ssquare_conv1/BiasAdd?
c4net/ssquare_conv1/ReluRelu$c4net/ssquare_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
c4net/ssquare_conv1/Relu?
%c4net/rcol_conv/Conv2D/ReadVariableOpReadVariableOp.c4net_rcol_conv_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02'
%c4net/rcol_conv/Conv2D/ReadVariableOp?
c4net/rcol_conv/Conv2DConv2Dinput-c4net/rcol_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
c4net/rcol_conv/Conv2D?
&c4net/rcol_conv/BiasAdd/ReadVariableOpReadVariableOp/c4net_rcol_conv_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02(
&c4net/rcol_conv/BiasAdd/ReadVariableOp?
c4net/rcol_conv/BiasAddBiasAddc4net/rcol_conv/Conv2D:output:0.c4net/rcol_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
c4net/rcol_conv/BiasAdd?
c4net/rcol_conv/ReluRelu c4net/rcol_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
c4net/rcol_conv/Relu?
%c4net/rrow_conv/Conv2D/ReadVariableOpReadVariableOp.c4net_rrow_conv_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02'
%c4net/rrow_conv/Conv2D/ReadVariableOp?
c4net/rrow_conv/Conv2DConv2Dinput-c4net/rrow_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
c4net/rrow_conv/Conv2D?
&c4net/rrow_conv/BiasAdd/ReadVariableOpReadVariableOp/c4net_rrow_conv_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02(
&c4net/rrow_conv/BiasAdd/ReadVariableOp?
c4net/rrow_conv/BiasAddBiasAddc4net/rrow_conv/Conv2D:output:0.c4net/rrow_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
c4net/rrow_conv/BiasAdd?
c4net/rrow_conv/ReluRelu c4net/rrow_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
c4net/rrow_conv/Relu?
&c4net/rcol_conv2/Conv2D/ReadVariableOpReadVariableOp/c4net_rcol_conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02(
&c4net/rcol_conv2/Conv2D/ReadVariableOp?
c4net/rcol_conv2/Conv2DConv2D&c4net/rsquare_conv1/Relu:activations:0.c4net/rcol_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
c4net/rcol_conv2/Conv2D?
'c4net/rcol_conv2/BiasAdd/ReadVariableOpReadVariableOp0c4net_rcol_conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'c4net/rcol_conv2/BiasAdd/ReadVariableOp?
c4net/rcol_conv2/BiasAddBiasAdd c4net/rcol_conv2/Conv2D:output:0/c4net/rcol_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
c4net/rcol_conv2/BiasAdd?
c4net/rcol_conv2/ReluRelu!c4net/rcol_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
c4net/rcol_conv2/Relu?
&c4net/rrow_conv2/Conv2D/ReadVariableOpReadVariableOp/c4net_rrow_conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02(
&c4net/rrow_conv2/Conv2D/ReadVariableOp?
c4net/rrow_conv2/Conv2DConv2D&c4net/rsquare_conv1/Relu:activations:0.c4net/rrow_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
c4net/rrow_conv2/Conv2D?
'c4net/rrow_conv2/BiasAdd/ReadVariableOpReadVariableOp0c4net_rrow_conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'c4net/rrow_conv2/BiasAdd/ReadVariableOp?
c4net/rrow_conv2/BiasAddBiasAdd c4net/rrow_conv2/Conv2D:output:0/c4net/rrow_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
c4net/rrow_conv2/BiasAdd?
c4net/rrow_conv2/ReluRelu!c4net/rrow_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
c4net/rrow_conv2/Relu?
)c4net/rsquare_conv2/Conv2D/ReadVariableOpReadVariableOp2c4net_rsquare_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)c4net/rsquare_conv2/Conv2D/ReadVariableOp?
c4net/rsquare_conv2/Conv2DConv2D&c4net/rsquare_conv1/Relu:activations:01c4net/rsquare_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
c4net/rsquare_conv2/Conv2D?
*c4net/rsquare_conv2/BiasAdd/ReadVariableOpReadVariableOp3c4net_rsquare_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*c4net/rsquare_conv2/BiasAdd/ReadVariableOp?
c4net/rsquare_conv2/BiasAddBiasAdd#c4net/rsquare_conv2/Conv2D:output:02c4net/rsquare_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
c4net/rsquare_conv2/BiasAdd?
c4net/rsquare_conv2/ReluRelu$c4net/rsquare_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
c4net/rsquare_conv2/Relu?
%c4net/scol_conv/Conv2D/ReadVariableOpReadVariableOp.c4net_scol_conv_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02'
%c4net/scol_conv/Conv2D/ReadVariableOp?
c4net/scol_conv/Conv2DConv2Dinput-c4net/scol_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
c4net/scol_conv/Conv2D?
&c4net/scol_conv/BiasAdd/ReadVariableOpReadVariableOp/c4net_scol_conv_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02(
&c4net/scol_conv/BiasAdd/ReadVariableOp?
c4net/scol_conv/BiasAddBiasAddc4net/scol_conv/Conv2D:output:0.c4net/scol_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
c4net/scol_conv/BiasAdd?
c4net/scol_conv/ReluRelu c4net/scol_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
c4net/scol_conv/Relu?
%c4net/srow_conv/Conv2D/ReadVariableOpReadVariableOp.c4net_srow_conv_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02'
%c4net/srow_conv/Conv2D/ReadVariableOp?
c4net/srow_conv/Conv2DConv2Dinput-c4net/srow_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
c4net/srow_conv/Conv2D?
&c4net/srow_conv/BiasAdd/ReadVariableOpReadVariableOp/c4net_srow_conv_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02(
&c4net/srow_conv/BiasAdd/ReadVariableOp?
c4net/srow_conv/BiasAddBiasAddc4net/srow_conv/Conv2D:output:0.c4net/srow_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
c4net/srow_conv/BiasAdd?
c4net/srow_conv/ReluRelu c4net/srow_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
c4net/srow_conv/Relu?
&c4net/scol_conv2/Conv2D/ReadVariableOpReadVariableOp/c4net_scol_conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02(
&c4net/scol_conv2/Conv2D/ReadVariableOp?
c4net/scol_conv2/Conv2DConv2D&c4net/ssquare_conv1/Relu:activations:0.c4net/scol_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
c4net/scol_conv2/Conv2D?
'c4net/scol_conv2/BiasAdd/ReadVariableOpReadVariableOp0c4net_scol_conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'c4net/scol_conv2/BiasAdd/ReadVariableOp?
c4net/scol_conv2/BiasAddBiasAdd c4net/scol_conv2/Conv2D:output:0/c4net/scol_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
c4net/scol_conv2/BiasAdd?
c4net/scol_conv2/ReluRelu!c4net/scol_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
c4net/scol_conv2/Relu?
&c4net/srow_conv2/Conv2D/ReadVariableOpReadVariableOp/c4net_srow_conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02(
&c4net/srow_conv2/Conv2D/ReadVariableOp?
c4net/srow_conv2/Conv2DConv2D&c4net/ssquare_conv1/Relu:activations:0.c4net/srow_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
c4net/srow_conv2/Conv2D?
'c4net/srow_conv2/BiasAdd/ReadVariableOpReadVariableOp0c4net_srow_conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'c4net/srow_conv2/BiasAdd/ReadVariableOp?
c4net/srow_conv2/BiasAddBiasAdd c4net/srow_conv2/Conv2D:output:0/c4net/srow_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
c4net/srow_conv2/BiasAdd?
c4net/srow_conv2/ReluRelu!c4net/srow_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
c4net/srow_conv2/Relu?
)c4net/ssquare_conv2/Conv2D/ReadVariableOpReadVariableOp2c4net_ssquare_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)c4net/ssquare_conv2/Conv2D/ReadVariableOp?
c4net/ssquare_conv2/Conv2DConv2D&c4net/ssquare_conv1/Relu:activations:01c4net/ssquare_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
c4net/ssquare_conv2/Conv2D?
*c4net/ssquare_conv2/BiasAdd/ReadVariableOpReadVariableOp3c4net_ssquare_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*c4net/ssquare_conv2/BiasAdd/ReadVariableOp?
c4net/ssquare_conv2/BiasAddBiasAdd#c4net/ssquare_conv2/Conv2D:output:02c4net/ssquare_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
c4net/ssquare_conv2/BiasAdd?
c4net/ssquare_conv2/ReluRelu$c4net/ssquare_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
c4net/ssquare_conv2/Relu
c4net/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
c4net/flatten_7/Const?
c4net/flatten_7/ReshapeReshape&c4net/rsquare_conv2/Relu:activations:0c4net/flatten_7/Const:output:0*
T0*'
_output_shapes
:?????????(2
c4net/flatten_7/Reshape
c4net/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
c4net/flatten_8/Const?
c4net/flatten_8/ReshapeReshape#c4net/rrow_conv2/Relu:activations:0c4net/flatten_8/Const:output:0*
T0*'
_output_shapes
:?????????2
c4net/flatten_8/Reshape
c4net/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
c4net/flatten_9/Const?
c4net/flatten_9/ReshapeReshape#c4net/rcol_conv2/Relu:activations:0c4net/flatten_9/Const:output:0*
T0*'
_output_shapes
:?????????2
c4net/flatten_9/Reshape?
c4net/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????<   2
c4net/flatten_10/Const?
c4net/flatten_10/ReshapeReshape"c4net/rrow_conv/Relu:activations:0c4net/flatten_10/Const:output:0*
T0*'
_output_shapes
:?????????<2
c4net/flatten_10/Reshape?
c4net/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????F   2
c4net/flatten_11/Const?
c4net/flatten_11/ReshapeReshape"c4net/rcol_conv/Relu:activations:0c4net/flatten_11/Const:output:0*
T0*'
_output_shapes
:?????????F2
c4net/flatten_11/Reshape
c4net/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
c4net/flatten_1/Const?
c4net/flatten_1/ReshapeReshape&c4net/ssquare_conv2/Relu:activations:0c4net/flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????(2
c4net/flatten_1/Reshape
c4net/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
c4net/flatten_2/Const?
c4net/flatten_2/ReshapeReshape#c4net/srow_conv2/Relu:activations:0c4net/flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????2
c4net/flatten_2/Reshape
c4net/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
c4net/flatten_3/Const?
c4net/flatten_3/ReshapeReshape#c4net/scol_conv2/Relu:activations:0c4net/flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????2
c4net/flatten_3/Reshape
c4net/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????<   2
c4net/flatten_4/Const?
c4net/flatten_4/ReshapeReshape"c4net/srow_conv/Relu:activations:0c4net/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????<2
c4net/flatten_4/Reshape
c4net/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????F   2
c4net/flatten_5/Const?
c4net/flatten_5/ReshapeReshape"c4net/scol_conv/Relu:activations:0c4net/flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????F2
c4net/flatten_5/Reshape?
c4net/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
c4net/concatenate_1/concat/axis?
c4net/concatenate_1/concatConcatV2 c4net/flatten_7/Reshape:output:0 c4net/flatten_8/Reshape:output:0 c4net/flatten_9/Reshape:output:0!c4net/flatten_10/Reshape:output:0!c4net/flatten_11/Reshape:output:0(c4net/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
c4net/concatenate_1/concat?
c4net/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
c4net/concatenate/concat/axis?
c4net/concatenate/concatConcatV2 c4net/flatten_1/Reshape:output:0 c4net/flatten_2/Reshape:output:0 c4net/flatten_3/Reshape:output:0 c4net/flatten_4/Reshape:output:0 c4net/flatten_5/Reshape:output:0&c4net/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
c4net/concatenate/concat?
(c4net/rdense_layer/MatMul/ReadVariableOpReadVariableOp1c4net_rdense_layer_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02*
(c4net/rdense_layer/MatMul/ReadVariableOp?
c4net/rdense_layer/MatMulMatMul#c4net/concatenate_1/concat:output:00c4net/rdense_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
c4net/rdense_layer/MatMul?
)c4net/rdense_layer/BiasAdd/ReadVariableOpReadVariableOp2c4net_rdense_layer_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02+
)c4net/rdense_layer/BiasAdd/ReadVariableOp?
c4net/rdense_layer/BiasAddBiasAdd#c4net/rdense_layer/MatMul:product:01c4net/rdense_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
c4net/rdense_layer/BiasAdd?
c4net/rdense_layer/ReluRelu#c4net/rdense_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
c4net/rdense_layer/Relu?
(c4net/sdense_layer/MatMul/ReadVariableOpReadVariableOp1c4net_sdense_layer_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02*
(c4net/sdense_layer/MatMul/ReadVariableOp?
c4net/sdense_layer/MatMulMatMul!c4net/concatenate/concat:output:00c4net/sdense_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
c4net/sdense_layer/MatMul?
)c4net/sdense_layer/BiasAdd/ReadVariableOpReadVariableOp2c4net_sdense_layer_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02+
)c4net/sdense_layer/BiasAdd/ReadVariableOp?
c4net/sdense_layer/BiasAddBiasAdd#c4net/sdense_layer/MatMul:product:01c4net/sdense_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
c4net/sdense_layer/BiasAdd?
c4net/sdense_layer/ReluRelu#c4net/sdense_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
c4net/sdense_layer/Relu?
)c4net/result_output/MatMul/ReadVariableOpReadVariableOp2c4net_result_output_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02+
)c4net/result_output/MatMul/ReadVariableOp?
c4net/result_output/MatMulMatMul%c4net/rdense_layer/Relu:activations:01c4net/result_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
c4net/result_output/MatMul?
*c4net/result_output/BiasAdd/ReadVariableOpReadVariableOp3c4net_result_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*c4net/result_output/BiasAdd/ReadVariableOp?
c4net/result_output/BiasAddBiasAdd$c4net/result_output/MatMul:product:02c4net/result_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
c4net/result_output/BiasAdd?
c4net/result_output/SigmoidSigmoid$c4net/result_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
c4net/result_output/Sigmoid?
(c4net/score_output/MatMul/ReadVariableOpReadVariableOp1c4net_score_output_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02*
(c4net/score_output/MatMul/ReadVariableOp?
c4net/score_output/MatMulMatMul%c4net/sdense_layer/Relu:activations:00c4net/score_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
c4net/score_output/MatMul?
)c4net/score_output/BiasAdd/ReadVariableOpReadVariableOp2c4net_score_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)c4net/score_output/BiasAdd/ReadVariableOp?
c4net/score_output/BiasAddBiasAdd#c4net/score_output/MatMul:product:01c4net/score_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
c4net/score_output/BiasAdd?
c4net/score_output/SoftmaxSoftmax#c4net/score_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
c4net/score_output/Softmax?
IdentityIdentityc4net/result_output/Sigmoid:y:0'^c4net/rcol_conv/BiasAdd/ReadVariableOp&^c4net/rcol_conv/Conv2D/ReadVariableOp(^c4net/rcol_conv2/BiasAdd/ReadVariableOp'^c4net/rcol_conv2/Conv2D/ReadVariableOp*^c4net/rdense_layer/BiasAdd/ReadVariableOp)^c4net/rdense_layer/MatMul/ReadVariableOp+^c4net/result_output/BiasAdd/ReadVariableOp*^c4net/result_output/MatMul/ReadVariableOp'^c4net/rrow_conv/BiasAdd/ReadVariableOp&^c4net/rrow_conv/Conv2D/ReadVariableOp(^c4net/rrow_conv2/BiasAdd/ReadVariableOp'^c4net/rrow_conv2/Conv2D/ReadVariableOp+^c4net/rsquare_conv1/BiasAdd/ReadVariableOp*^c4net/rsquare_conv1/Conv2D/ReadVariableOp+^c4net/rsquare_conv2/BiasAdd/ReadVariableOp*^c4net/rsquare_conv2/Conv2D/ReadVariableOp'^c4net/scol_conv/BiasAdd/ReadVariableOp&^c4net/scol_conv/Conv2D/ReadVariableOp(^c4net/scol_conv2/BiasAdd/ReadVariableOp'^c4net/scol_conv2/Conv2D/ReadVariableOp*^c4net/score_output/BiasAdd/ReadVariableOp)^c4net/score_output/MatMul/ReadVariableOp*^c4net/sdense_layer/BiasAdd/ReadVariableOp)^c4net/sdense_layer/MatMul/ReadVariableOp'^c4net/srow_conv/BiasAdd/ReadVariableOp&^c4net/srow_conv/Conv2D/ReadVariableOp(^c4net/srow_conv2/BiasAdd/ReadVariableOp'^c4net/srow_conv2/Conv2D/ReadVariableOp+^c4net/ssquare_conv1/BiasAdd/ReadVariableOp*^c4net/ssquare_conv1/Conv2D/ReadVariableOp+^c4net/ssquare_conv2/BiasAdd/ReadVariableOp*^c4net/ssquare_conv2/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity$c4net/score_output/Softmax:softmax:0'^c4net/rcol_conv/BiasAdd/ReadVariableOp&^c4net/rcol_conv/Conv2D/ReadVariableOp(^c4net/rcol_conv2/BiasAdd/ReadVariableOp'^c4net/rcol_conv2/Conv2D/ReadVariableOp*^c4net/rdense_layer/BiasAdd/ReadVariableOp)^c4net/rdense_layer/MatMul/ReadVariableOp+^c4net/result_output/BiasAdd/ReadVariableOp*^c4net/result_output/MatMul/ReadVariableOp'^c4net/rrow_conv/BiasAdd/ReadVariableOp&^c4net/rrow_conv/Conv2D/ReadVariableOp(^c4net/rrow_conv2/BiasAdd/ReadVariableOp'^c4net/rrow_conv2/Conv2D/ReadVariableOp+^c4net/rsquare_conv1/BiasAdd/ReadVariableOp*^c4net/rsquare_conv1/Conv2D/ReadVariableOp+^c4net/rsquare_conv2/BiasAdd/ReadVariableOp*^c4net/rsquare_conv2/Conv2D/ReadVariableOp'^c4net/scol_conv/BiasAdd/ReadVariableOp&^c4net/scol_conv/Conv2D/ReadVariableOp(^c4net/scol_conv2/BiasAdd/ReadVariableOp'^c4net/scol_conv2/Conv2D/ReadVariableOp*^c4net/score_output/BiasAdd/ReadVariableOp)^c4net/score_output/MatMul/ReadVariableOp*^c4net/sdense_layer/BiasAdd/ReadVariableOp)^c4net/sdense_layer/MatMul/ReadVariableOp'^c4net/srow_conv/BiasAdd/ReadVariableOp&^c4net/srow_conv/Conv2D/ReadVariableOp(^c4net/srow_conv2/BiasAdd/ReadVariableOp'^c4net/srow_conv2/Conv2D/ReadVariableOp+^c4net/ssquare_conv1/BiasAdd/ReadVariableOp*^c4net/ssquare_conv1/Conv2D/ReadVariableOp+^c4net/ssquare_conv2/BiasAdd/ReadVariableOp*^c4net/ssquare_conv2/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&c4net/rcol_conv/BiasAdd/ReadVariableOp&c4net/rcol_conv/BiasAdd/ReadVariableOp2N
%c4net/rcol_conv/Conv2D/ReadVariableOp%c4net/rcol_conv/Conv2D/ReadVariableOp2R
'c4net/rcol_conv2/BiasAdd/ReadVariableOp'c4net/rcol_conv2/BiasAdd/ReadVariableOp2P
&c4net/rcol_conv2/Conv2D/ReadVariableOp&c4net/rcol_conv2/Conv2D/ReadVariableOp2V
)c4net/rdense_layer/BiasAdd/ReadVariableOp)c4net/rdense_layer/BiasAdd/ReadVariableOp2T
(c4net/rdense_layer/MatMul/ReadVariableOp(c4net/rdense_layer/MatMul/ReadVariableOp2X
*c4net/result_output/BiasAdd/ReadVariableOp*c4net/result_output/BiasAdd/ReadVariableOp2V
)c4net/result_output/MatMul/ReadVariableOp)c4net/result_output/MatMul/ReadVariableOp2P
&c4net/rrow_conv/BiasAdd/ReadVariableOp&c4net/rrow_conv/BiasAdd/ReadVariableOp2N
%c4net/rrow_conv/Conv2D/ReadVariableOp%c4net/rrow_conv/Conv2D/ReadVariableOp2R
'c4net/rrow_conv2/BiasAdd/ReadVariableOp'c4net/rrow_conv2/BiasAdd/ReadVariableOp2P
&c4net/rrow_conv2/Conv2D/ReadVariableOp&c4net/rrow_conv2/Conv2D/ReadVariableOp2X
*c4net/rsquare_conv1/BiasAdd/ReadVariableOp*c4net/rsquare_conv1/BiasAdd/ReadVariableOp2V
)c4net/rsquare_conv1/Conv2D/ReadVariableOp)c4net/rsquare_conv1/Conv2D/ReadVariableOp2X
*c4net/rsquare_conv2/BiasAdd/ReadVariableOp*c4net/rsquare_conv2/BiasAdd/ReadVariableOp2V
)c4net/rsquare_conv2/Conv2D/ReadVariableOp)c4net/rsquare_conv2/Conv2D/ReadVariableOp2P
&c4net/scol_conv/BiasAdd/ReadVariableOp&c4net/scol_conv/BiasAdd/ReadVariableOp2N
%c4net/scol_conv/Conv2D/ReadVariableOp%c4net/scol_conv/Conv2D/ReadVariableOp2R
'c4net/scol_conv2/BiasAdd/ReadVariableOp'c4net/scol_conv2/BiasAdd/ReadVariableOp2P
&c4net/scol_conv2/Conv2D/ReadVariableOp&c4net/scol_conv2/Conv2D/ReadVariableOp2V
)c4net/score_output/BiasAdd/ReadVariableOp)c4net/score_output/BiasAdd/ReadVariableOp2T
(c4net/score_output/MatMul/ReadVariableOp(c4net/score_output/MatMul/ReadVariableOp2V
)c4net/sdense_layer/BiasAdd/ReadVariableOp)c4net/sdense_layer/BiasAdd/ReadVariableOp2T
(c4net/sdense_layer/MatMul/ReadVariableOp(c4net/sdense_layer/MatMul/ReadVariableOp2P
&c4net/srow_conv/BiasAdd/ReadVariableOp&c4net/srow_conv/BiasAdd/ReadVariableOp2N
%c4net/srow_conv/Conv2D/ReadVariableOp%c4net/srow_conv/Conv2D/ReadVariableOp2R
'c4net/srow_conv2/BiasAdd/ReadVariableOp'c4net/srow_conv2/BiasAdd/ReadVariableOp2P
&c4net/srow_conv2/Conv2D/ReadVariableOp&c4net/srow_conv2/Conv2D/ReadVariableOp2X
*c4net/ssquare_conv1/BiasAdd/ReadVariableOp*c4net/ssquare_conv1/BiasAdd/ReadVariableOp2V
)c4net/ssquare_conv1/Conv2D/ReadVariableOp)c4net/ssquare_conv1/Conv2D/ReadVariableOp2X
*c4net/ssquare_conv2/BiasAdd/ReadVariableOp*c4net/ssquare_conv2/BiasAdd/ReadVariableOp2V
)c4net/ssquare_conv2/Conv2D/ReadVariableOp)c4net/ssquare_conv2/Conv2D/ReadVariableOp:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
H__inference_ssquare_conv2_layer_call_and_return_conditional_losses_27266

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_concatenate_layer_call_and_return_conditional_losses_25833

inputs
inputs_1
inputs_2
inputs_3
inputs_4
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????(:?????????:?????????:?????????<:?????????F:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????<
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
%__inference_c4net_layer_call_fn_27215

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:

	unknown_4:
#
	unknown_5:

	unknown_6:
#
	unknown_7:

	unknown_8:
#
	unknown_9:


unknown_10:
$

unknown_11:

unknown_12:$

unknown_13:


unknown_14:
$

unknown_15:


unknown_16:
$

unknown_17:


unknown_18:
$

unknown_19:


unknown_20:
$

unknown_21:

unknown_22:

unknown_23:	?d

unknown_24:d

unknown_25:	?d

unknown_26:d

unknown_27:d

unknown_28:

unknown_29:d

unknown_30:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_c4net_layer_call_and_return_conditional_losses_263842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_rsquare_conv1_layer_call_and_return_conditional_losses_25538

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_27594
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????(:?????????:?????????:?????????<:?????????F:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????<
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????F
"
_user_specified_name
inputs/4
?
?
E__inference_scol_conv2_layer_call_and_return_conditional_losses_27306

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_score_output_layer_call_and_return_conditional_losses_25897

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
a
E__inference_flatten_10_layer_call_and_return_conditional_losses_27549

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????<   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????<2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
D__inference_srow_conv_layer_call_and_return_conditional_losses_27326

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
H__inference_result_output_layer_call_and_return_conditional_losses_25880

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
)__inference_scol_conv_layer_call_fn_27355

inputs!
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_scol_conv_layer_call_and_return_conditional_losses_256572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_3_layer_call_fn_27488

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_257932
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
ޅ
?
@__inference_c4net_layer_call_and_return_conditional_losses_26384

inputs-
rsquare_conv1_26290:!
rsquare_conv1_26292:-
ssquare_conv1_26295:!
ssquare_conv1_26297:)
rcol_conv_26300:

rcol_conv_26302:
)
rrow_conv_26305:

rrow_conv_26307:
*
rcol_conv2_26310:

rcol_conv2_26312:
*
rrow_conv2_26315:

rrow_conv2_26317:
-
rsquare_conv2_26320:!
rsquare_conv2_26322:)
scol_conv_26325:

scol_conv_26327:
)
srow_conv_26330:

srow_conv_26332:
*
scol_conv2_26335:

scol_conv2_26337:
*
srow_conv2_26340:

srow_conv2_26342:
-
ssquare_conv2_26345:!
ssquare_conv2_26347:%
rdense_layer_26362:	?d 
rdense_layer_26364:d%
sdense_layer_26367:	?d 
sdense_layer_26369:d%
result_output_26372:d!
result_output_26374:$
score_output_26377:d 
score_output_26379:
identity

identity_1??!rcol_conv/StatefulPartitionedCall?"rcol_conv2/StatefulPartitionedCall?$rdense_layer/StatefulPartitionedCall?%result_output/StatefulPartitionedCall?!rrow_conv/StatefulPartitionedCall?"rrow_conv2/StatefulPartitionedCall?%rsquare_conv1/StatefulPartitionedCall?%rsquare_conv2/StatefulPartitionedCall?!scol_conv/StatefulPartitionedCall?"scol_conv2/StatefulPartitionedCall?$score_output/StatefulPartitionedCall?$sdense_layer/StatefulPartitionedCall?!srow_conv/StatefulPartitionedCall?"srow_conv2/StatefulPartitionedCall?%ssquare_conv1/StatefulPartitionedCall?%ssquare_conv2/StatefulPartitionedCall?
%rsquare_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsrsquare_conv1_26290rsquare_conv1_26292*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rsquare_conv1_layer_call_and_return_conditional_losses_255382'
%rsquare_conv1/StatefulPartitionedCall?
%ssquare_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsssquare_conv1_26295ssquare_conv1_26297*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_ssquare_conv1_layer_call_and_return_conditional_losses_255552'
%ssquare_conv1/StatefulPartitionedCall?
!rcol_conv/StatefulPartitionedCallStatefulPartitionedCallinputsrcol_conv_26300rcol_conv_26302*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_rcol_conv_layer_call_and_return_conditional_losses_255722#
!rcol_conv/StatefulPartitionedCall?
!rrow_conv/StatefulPartitionedCallStatefulPartitionedCallinputsrrow_conv_26305rrow_conv_26307*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_rrow_conv_layer_call_and_return_conditional_losses_255892#
!rrow_conv/StatefulPartitionedCall?
"rcol_conv2/StatefulPartitionedCallStatefulPartitionedCall.rsquare_conv1/StatefulPartitionedCall:output:0rcol_conv2_26310rcol_conv2_26312*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_rcol_conv2_layer_call_and_return_conditional_losses_256062$
"rcol_conv2/StatefulPartitionedCall?
"rrow_conv2/StatefulPartitionedCallStatefulPartitionedCall.rsquare_conv1/StatefulPartitionedCall:output:0rrow_conv2_26315rrow_conv2_26317*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_rrow_conv2_layer_call_and_return_conditional_losses_256232$
"rrow_conv2/StatefulPartitionedCall?
%rsquare_conv2/StatefulPartitionedCallStatefulPartitionedCall.rsquare_conv1/StatefulPartitionedCall:output:0rsquare_conv2_26320rsquare_conv2_26322*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rsquare_conv2_layer_call_and_return_conditional_losses_256402'
%rsquare_conv2/StatefulPartitionedCall?
!scol_conv/StatefulPartitionedCallStatefulPartitionedCallinputsscol_conv_26325scol_conv_26327*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_scol_conv_layer_call_and_return_conditional_losses_256572#
!scol_conv/StatefulPartitionedCall?
!srow_conv/StatefulPartitionedCallStatefulPartitionedCallinputssrow_conv_26330srow_conv_26332*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_srow_conv_layer_call_and_return_conditional_losses_256742#
!srow_conv/StatefulPartitionedCall?
"scol_conv2/StatefulPartitionedCallStatefulPartitionedCall.ssquare_conv1/StatefulPartitionedCall:output:0scol_conv2_26335scol_conv2_26337*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_scol_conv2_layer_call_and_return_conditional_losses_256912$
"scol_conv2/StatefulPartitionedCall?
"srow_conv2/StatefulPartitionedCallStatefulPartitionedCall.ssquare_conv1/StatefulPartitionedCall:output:0srow_conv2_26340srow_conv2_26342*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_srow_conv2_layer_call_and_return_conditional_losses_257082$
"srow_conv2/StatefulPartitionedCall?
%ssquare_conv2/StatefulPartitionedCallStatefulPartitionedCall.ssquare_conv1/StatefulPartitionedCall:output:0ssquare_conv2_26345ssquare_conv2_26347*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_ssquare_conv2_layer_call_and_return_conditional_losses_257252'
%ssquare_conv2/StatefulPartitionedCall?
flatten_7/PartitionedCallPartitionedCall.rsquare_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_257372
flatten_7/PartitionedCall?
flatten_8/PartitionedCallPartitionedCall+rrow_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_257452
flatten_8/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall+rcol_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_9_layer_call_and_return_conditional_losses_257532
flatten_9/PartitionedCall?
flatten_10/PartitionedCallPartitionedCall*rrow_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_10_layer_call_and_return_conditional_losses_257612
flatten_10/PartitionedCall?
flatten_11/PartitionedCallPartitionedCall*rcol_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_11_layer_call_and_return_conditional_losses_257692
flatten_11/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall.ssquare_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_257772
flatten_1/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall+srow_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_257852
flatten_2/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall+scol_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_257932
flatten_3/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall*srow_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_258012
flatten_4/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall*scol_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_258092
flatten_5/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall"flatten_7/PartitionedCall:output:0"flatten_8/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0#flatten_10/PartitionedCall:output:0#flatten_11/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_258212
concatenate_1/PartitionedCall?
concatenate/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_258332
concatenate/PartitionedCall?
$rdense_layer/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0rdense_layer_26362rdense_layer_26364*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_rdense_layer_layer_call_and_return_conditional_losses_258462&
$rdense_layer/StatefulPartitionedCall?
$sdense_layer/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sdense_layer_26367sdense_layer_26369*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sdense_layer_layer_call_and_return_conditional_losses_258632&
$sdense_layer/StatefulPartitionedCall?
%result_output/StatefulPartitionedCallStatefulPartitionedCall-rdense_layer/StatefulPartitionedCall:output:0result_output_26372result_output_26374*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_result_output_layer_call_and_return_conditional_losses_258802'
%result_output/StatefulPartitionedCall?
$score_output/StatefulPartitionedCallStatefulPartitionedCall-sdense_layer/StatefulPartitionedCall:output:0score_output_26377score_output_26379*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_score_output_layer_call_and_return_conditional_losses_258972&
$score_output/StatefulPartitionedCall?
IdentityIdentity-score_output/StatefulPartitionedCall:output:0"^rcol_conv/StatefulPartitionedCall#^rcol_conv2/StatefulPartitionedCall%^rdense_layer/StatefulPartitionedCall&^result_output/StatefulPartitionedCall"^rrow_conv/StatefulPartitionedCall#^rrow_conv2/StatefulPartitionedCall&^rsquare_conv1/StatefulPartitionedCall&^rsquare_conv2/StatefulPartitionedCall"^scol_conv/StatefulPartitionedCall#^scol_conv2/StatefulPartitionedCall%^score_output/StatefulPartitionedCall%^sdense_layer/StatefulPartitionedCall"^srow_conv/StatefulPartitionedCall#^srow_conv2/StatefulPartitionedCall&^ssquare_conv1/StatefulPartitionedCall&^ssquare_conv2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity.result_output/StatefulPartitionedCall:output:0"^rcol_conv/StatefulPartitionedCall#^rcol_conv2/StatefulPartitionedCall%^rdense_layer/StatefulPartitionedCall&^result_output/StatefulPartitionedCall"^rrow_conv/StatefulPartitionedCall#^rrow_conv2/StatefulPartitionedCall&^rsquare_conv1/StatefulPartitionedCall&^rsquare_conv2/StatefulPartitionedCall"^scol_conv/StatefulPartitionedCall#^scol_conv2/StatefulPartitionedCall%^score_output/StatefulPartitionedCall%^sdense_layer/StatefulPartitionedCall"^srow_conv/StatefulPartitionedCall#^srow_conv2/StatefulPartitionedCall&^ssquare_conv1/StatefulPartitionedCall&^ssquare_conv2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!rcol_conv/StatefulPartitionedCall!rcol_conv/StatefulPartitionedCall2H
"rcol_conv2/StatefulPartitionedCall"rcol_conv2/StatefulPartitionedCall2L
$rdense_layer/StatefulPartitionedCall$rdense_layer/StatefulPartitionedCall2N
%result_output/StatefulPartitionedCall%result_output/StatefulPartitionedCall2F
!rrow_conv/StatefulPartitionedCall!rrow_conv/StatefulPartitionedCall2H
"rrow_conv2/StatefulPartitionedCall"rrow_conv2/StatefulPartitionedCall2N
%rsquare_conv1/StatefulPartitionedCall%rsquare_conv1/StatefulPartitionedCall2N
%rsquare_conv2/StatefulPartitionedCall%rsquare_conv2/StatefulPartitionedCall2F
!scol_conv/StatefulPartitionedCall!scol_conv/StatefulPartitionedCall2H
"scol_conv2/StatefulPartitionedCall"scol_conv2/StatefulPartitionedCall2L
$score_output/StatefulPartitionedCall$score_output/StatefulPartitionedCall2L
$sdense_layer/StatefulPartitionedCall$sdense_layer/StatefulPartitionedCall2F
!srow_conv/StatefulPartitionedCall!srow_conv/StatefulPartitionedCall2H
"srow_conv2/StatefulPartitionedCall"srow_conv2/StatefulPartitionedCall2N
%ssquare_conv1/StatefulPartitionedCall%ssquare_conv1/StatefulPartitionedCall2N
%ssquare_conv2/StatefulPartitionedCall%ssquare_conv2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
H__inference_result_output_layer_call_and_return_conditional_losses_27674

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
E
)__inference_flatten_8_layer_call_fn_27532

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_257452
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
??
?
!__inference__traced_restore_27993
file_prefix?
%assignvariableop_ssquare_conv1_kernel:3
%assignvariableop_1_ssquare_conv1_bias:A
'assignvariableop_2_rsquare_conv1_kernel:3
%assignvariableop_3_rsquare_conv1_bias:A
'assignvariableop_4_ssquare_conv2_kernel:3
%assignvariableop_5_ssquare_conv2_bias:>
$assignvariableop_6_srow_conv2_kernel:
0
"assignvariableop_7_srow_conv2_bias:
>
$assignvariableop_8_scol_conv2_kernel:
0
"assignvariableop_9_scol_conv2_bias:
>
$assignvariableop_10_srow_conv_kernel:
0
"assignvariableop_11_srow_conv_bias:
>
$assignvariableop_12_scol_conv_kernel:
0
"assignvariableop_13_scol_conv_bias:
B
(assignvariableop_14_rsquare_conv2_kernel:4
&assignvariableop_15_rsquare_conv2_bias:?
%assignvariableop_16_rrow_conv2_kernel:
1
#assignvariableop_17_rrow_conv2_bias:
?
%assignvariableop_18_rcol_conv2_kernel:
1
#assignvariableop_19_rcol_conv2_bias:
>
$assignvariableop_20_rrow_conv_kernel:
0
"assignvariableop_21_rrow_conv_bias:
>
$assignvariableop_22_rcol_conv_kernel:
0
"assignvariableop_23_rcol_conv_bias:
:
'assignvariableop_24_sdense_layer_kernel:	?d3
%assignvariableop_25_sdense_layer_bias:d:
'assignvariableop_26_rdense_layer_kernel:	?d3
%assignvariableop_27_rdense_layer_bias:d9
'assignvariableop_28_score_output_kernel:d3
%assignvariableop_29_score_output_bias::
(assignvariableop_30_result_output_kernel:d4
&assignvariableop_31_result_output_bias:#
assignvariableop_32_total: #
assignvariableop_33_count: %
assignvariableop_34_total_1: %
assignvariableop_35_count_1: %
assignvariableop_36_total_2: %
assignvariableop_37_count_2: %
assignvariableop_38_total_3: %
assignvariableop_39_count_3: %
assignvariableop_40_total_4: %
assignvariableop_41_count_4: %
assignvariableop_42_total_5: %
assignvariableop_43_count_5: %
assignvariableop_44_total_6: %
assignvariableop_45_count_6: 
identity_47??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*?
value?B?/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp%assignvariableop_ssquare_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp%assignvariableop_1_ssquare_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp'assignvariableop_2_rsquare_conv1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp%assignvariableop_3_rsquare_conv1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp'assignvariableop_4_ssquare_conv2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp%assignvariableop_5_ssquare_conv2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_srow_conv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_srow_conv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_scol_conv2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_scol_conv2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_srow_conv_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_srow_conv_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_scol_conv_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_scol_conv_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp(assignvariableop_14_rsquare_conv2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp&assignvariableop_15_rsquare_conv2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_rrow_conv2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp#assignvariableop_17_rrow_conv2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_rcol_conv2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_rcol_conv2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp$assignvariableop_20_rrow_conv_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_rrow_conv_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp$assignvariableop_22_rcol_conv_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp"assignvariableop_23_rcol_conv_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_sdense_layer_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp%assignvariableop_25_sdense_layer_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_rdense_layer_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_rdense_layer_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp'assignvariableop_28_score_output_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp%assignvariableop_29_score_output_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_result_output_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp&assignvariableop_31_result_output_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_totalIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_countIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_total_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_count_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_total_2Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_count_2Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpassignvariableop_38_total_3Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpassignvariableop_39_count_3Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_total_4Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_count_4Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_total_5Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_count_5Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpassignvariableop_44_total_6Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpassignvariableop_45_count_6Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_459
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_46?
Identity_47IdentityIdentity_46:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_47"#
identity_47Identity_47:output:0*q
_input_shapes`
^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
*__inference_srow_conv2_layer_call_fn_27295

inputs!
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_srow_conv2_layer_call_and_return_conditional_losses_257082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_rcol_conv2_layer_call_and_return_conditional_losses_25606

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_scol_conv_layer_call_and_return_conditional_losses_25657

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_25785

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
H__inference_rsquare_conv1_layer_call_and_return_conditional_losses_27246

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_rrow_conv_layer_call_fn_27435

inputs!
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_rrow_conv_layer_call_and_return_conditional_losses_255892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_25801

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????<   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????<2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_27461

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????(2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
@__inference_c4net_layer_call_and_return_conditional_losses_26932

inputsF
,rsquare_conv1_conv2d_readvariableop_resource:;
-rsquare_conv1_biasadd_readvariableop_resource:F
,ssquare_conv1_conv2d_readvariableop_resource:;
-ssquare_conv1_biasadd_readvariableop_resource:B
(rcol_conv_conv2d_readvariableop_resource:
7
)rcol_conv_biasadd_readvariableop_resource:
B
(rrow_conv_conv2d_readvariableop_resource:
7
)rrow_conv_biasadd_readvariableop_resource:
C
)rcol_conv2_conv2d_readvariableop_resource:
8
*rcol_conv2_biasadd_readvariableop_resource:
C
)rrow_conv2_conv2d_readvariableop_resource:
8
*rrow_conv2_biasadd_readvariableop_resource:
F
,rsquare_conv2_conv2d_readvariableop_resource:;
-rsquare_conv2_biasadd_readvariableop_resource:B
(scol_conv_conv2d_readvariableop_resource:
7
)scol_conv_biasadd_readvariableop_resource:
B
(srow_conv_conv2d_readvariableop_resource:
7
)srow_conv_biasadd_readvariableop_resource:
C
)scol_conv2_conv2d_readvariableop_resource:
8
*scol_conv2_biasadd_readvariableop_resource:
C
)srow_conv2_conv2d_readvariableop_resource:
8
*srow_conv2_biasadd_readvariableop_resource:
F
,ssquare_conv2_conv2d_readvariableop_resource:;
-ssquare_conv2_biasadd_readvariableop_resource:>
+rdense_layer_matmul_readvariableop_resource:	?d:
,rdense_layer_biasadd_readvariableop_resource:d>
+sdense_layer_matmul_readvariableop_resource:	?d:
,sdense_layer_biasadd_readvariableop_resource:d>
,result_output_matmul_readvariableop_resource:d;
-result_output_biasadd_readvariableop_resource:=
+score_output_matmul_readvariableop_resource:d:
,score_output_biasadd_readvariableop_resource:
identity

identity_1?? rcol_conv/BiasAdd/ReadVariableOp?rcol_conv/Conv2D/ReadVariableOp?!rcol_conv2/BiasAdd/ReadVariableOp? rcol_conv2/Conv2D/ReadVariableOp?#rdense_layer/BiasAdd/ReadVariableOp?"rdense_layer/MatMul/ReadVariableOp?$result_output/BiasAdd/ReadVariableOp?#result_output/MatMul/ReadVariableOp? rrow_conv/BiasAdd/ReadVariableOp?rrow_conv/Conv2D/ReadVariableOp?!rrow_conv2/BiasAdd/ReadVariableOp? rrow_conv2/Conv2D/ReadVariableOp?$rsquare_conv1/BiasAdd/ReadVariableOp?#rsquare_conv1/Conv2D/ReadVariableOp?$rsquare_conv2/BiasAdd/ReadVariableOp?#rsquare_conv2/Conv2D/ReadVariableOp? scol_conv/BiasAdd/ReadVariableOp?scol_conv/Conv2D/ReadVariableOp?!scol_conv2/BiasAdd/ReadVariableOp? scol_conv2/Conv2D/ReadVariableOp?#score_output/BiasAdd/ReadVariableOp?"score_output/MatMul/ReadVariableOp?#sdense_layer/BiasAdd/ReadVariableOp?"sdense_layer/MatMul/ReadVariableOp? srow_conv/BiasAdd/ReadVariableOp?srow_conv/Conv2D/ReadVariableOp?!srow_conv2/BiasAdd/ReadVariableOp? srow_conv2/Conv2D/ReadVariableOp?$ssquare_conv1/BiasAdd/ReadVariableOp?#ssquare_conv1/Conv2D/ReadVariableOp?$ssquare_conv2/BiasAdd/ReadVariableOp?#ssquare_conv2/Conv2D/ReadVariableOp?
#rsquare_conv1/Conv2D/ReadVariableOpReadVariableOp,rsquare_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#rsquare_conv1/Conv2D/ReadVariableOp?
rsquare_conv1/Conv2DConv2Dinputs+rsquare_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
rsquare_conv1/Conv2D?
$rsquare_conv1/BiasAdd/ReadVariableOpReadVariableOp-rsquare_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$rsquare_conv1/BiasAdd/ReadVariableOp?
rsquare_conv1/BiasAddBiasAddrsquare_conv1/Conv2D:output:0,rsquare_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
rsquare_conv1/BiasAdd?
rsquare_conv1/ReluRelursquare_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
rsquare_conv1/Relu?
#ssquare_conv1/Conv2D/ReadVariableOpReadVariableOp,ssquare_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#ssquare_conv1/Conv2D/ReadVariableOp?
ssquare_conv1/Conv2DConv2Dinputs+ssquare_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ssquare_conv1/Conv2D?
$ssquare_conv1/BiasAdd/ReadVariableOpReadVariableOp-ssquare_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ssquare_conv1/BiasAdd/ReadVariableOp?
ssquare_conv1/BiasAddBiasAddssquare_conv1/Conv2D:output:0,ssquare_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ssquare_conv1/BiasAdd?
ssquare_conv1/ReluRelussquare_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
ssquare_conv1/Relu?
rcol_conv/Conv2D/ReadVariableOpReadVariableOp(rcol_conv_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02!
rcol_conv/Conv2D/ReadVariableOp?
rcol_conv/Conv2DConv2Dinputs'rcol_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
rcol_conv/Conv2D?
 rcol_conv/BiasAdd/ReadVariableOpReadVariableOp)rcol_conv_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 rcol_conv/BiasAdd/ReadVariableOp?
rcol_conv/BiasAddBiasAddrcol_conv/Conv2D:output:0(rcol_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
rcol_conv/BiasAdd~
rcol_conv/ReluRelurcol_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
rcol_conv/Relu?
rrow_conv/Conv2D/ReadVariableOpReadVariableOp(rrow_conv_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02!
rrow_conv/Conv2D/ReadVariableOp?
rrow_conv/Conv2DConv2Dinputs'rrow_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
rrow_conv/Conv2D?
 rrow_conv/BiasAdd/ReadVariableOpReadVariableOp)rrow_conv_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 rrow_conv/BiasAdd/ReadVariableOp?
rrow_conv/BiasAddBiasAddrrow_conv/Conv2D:output:0(rrow_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
rrow_conv/BiasAdd~
rrow_conv/ReluRelurrow_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
rrow_conv/Relu?
 rcol_conv2/Conv2D/ReadVariableOpReadVariableOp)rcol_conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02"
 rcol_conv2/Conv2D/ReadVariableOp?
rcol_conv2/Conv2DConv2D rsquare_conv1/Relu:activations:0(rcol_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
rcol_conv2/Conv2D?
!rcol_conv2/BiasAdd/ReadVariableOpReadVariableOp*rcol_conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02#
!rcol_conv2/BiasAdd/ReadVariableOp?
rcol_conv2/BiasAddBiasAddrcol_conv2/Conv2D:output:0)rcol_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
rcol_conv2/BiasAdd?
rcol_conv2/ReluRelurcol_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
rcol_conv2/Relu?
 rrow_conv2/Conv2D/ReadVariableOpReadVariableOp)rrow_conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02"
 rrow_conv2/Conv2D/ReadVariableOp?
rrow_conv2/Conv2DConv2D rsquare_conv1/Relu:activations:0(rrow_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
rrow_conv2/Conv2D?
!rrow_conv2/BiasAdd/ReadVariableOpReadVariableOp*rrow_conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02#
!rrow_conv2/BiasAdd/ReadVariableOp?
rrow_conv2/BiasAddBiasAddrrow_conv2/Conv2D:output:0)rrow_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
rrow_conv2/BiasAdd?
rrow_conv2/ReluRelurrow_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
rrow_conv2/Relu?
#rsquare_conv2/Conv2D/ReadVariableOpReadVariableOp,rsquare_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#rsquare_conv2/Conv2D/ReadVariableOp?
rsquare_conv2/Conv2DConv2D rsquare_conv1/Relu:activations:0+rsquare_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
rsquare_conv2/Conv2D?
$rsquare_conv2/BiasAdd/ReadVariableOpReadVariableOp-rsquare_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$rsquare_conv2/BiasAdd/ReadVariableOp?
rsquare_conv2/BiasAddBiasAddrsquare_conv2/Conv2D:output:0,rsquare_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
rsquare_conv2/BiasAdd?
rsquare_conv2/ReluRelursquare_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
rsquare_conv2/Relu?
scol_conv/Conv2D/ReadVariableOpReadVariableOp(scol_conv_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02!
scol_conv/Conv2D/ReadVariableOp?
scol_conv/Conv2DConv2Dinputs'scol_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
scol_conv/Conv2D?
 scol_conv/BiasAdd/ReadVariableOpReadVariableOp)scol_conv_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 scol_conv/BiasAdd/ReadVariableOp?
scol_conv/BiasAddBiasAddscol_conv/Conv2D:output:0(scol_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
scol_conv/BiasAdd~
scol_conv/ReluReluscol_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
scol_conv/Relu?
srow_conv/Conv2D/ReadVariableOpReadVariableOp(srow_conv_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02!
srow_conv/Conv2D/ReadVariableOp?
srow_conv/Conv2DConv2Dinputs'srow_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
srow_conv/Conv2D?
 srow_conv/BiasAdd/ReadVariableOpReadVariableOp)srow_conv_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 srow_conv/BiasAdd/ReadVariableOp?
srow_conv/BiasAddBiasAddsrow_conv/Conv2D:output:0(srow_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
srow_conv/BiasAdd~
srow_conv/ReluRelusrow_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
srow_conv/Relu?
 scol_conv2/Conv2D/ReadVariableOpReadVariableOp)scol_conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02"
 scol_conv2/Conv2D/ReadVariableOp?
scol_conv2/Conv2DConv2D ssquare_conv1/Relu:activations:0(scol_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
scol_conv2/Conv2D?
!scol_conv2/BiasAdd/ReadVariableOpReadVariableOp*scol_conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02#
!scol_conv2/BiasAdd/ReadVariableOp?
scol_conv2/BiasAddBiasAddscol_conv2/Conv2D:output:0)scol_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
scol_conv2/BiasAdd?
scol_conv2/ReluReluscol_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
scol_conv2/Relu?
 srow_conv2/Conv2D/ReadVariableOpReadVariableOp)srow_conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02"
 srow_conv2/Conv2D/ReadVariableOp?
srow_conv2/Conv2DConv2D ssquare_conv1/Relu:activations:0(srow_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
srow_conv2/Conv2D?
!srow_conv2/BiasAdd/ReadVariableOpReadVariableOp*srow_conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02#
!srow_conv2/BiasAdd/ReadVariableOp?
srow_conv2/BiasAddBiasAddsrow_conv2/Conv2D:output:0)srow_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
srow_conv2/BiasAdd?
srow_conv2/ReluRelusrow_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
srow_conv2/Relu?
#ssquare_conv2/Conv2D/ReadVariableOpReadVariableOp,ssquare_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#ssquare_conv2/Conv2D/ReadVariableOp?
ssquare_conv2/Conv2DConv2D ssquare_conv1/Relu:activations:0+ssquare_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ssquare_conv2/Conv2D?
$ssquare_conv2/BiasAdd/ReadVariableOpReadVariableOp-ssquare_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ssquare_conv2/BiasAdd/ReadVariableOp?
ssquare_conv2/BiasAddBiasAddssquare_conv2/Conv2D:output:0,ssquare_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ssquare_conv2/BiasAdd?
ssquare_conv2/ReluRelussquare_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
ssquare_conv2/Relus
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
flatten_7/Const?
flatten_7/ReshapeReshape rsquare_conv2/Relu:activations:0flatten_7/Const:output:0*
T0*'
_output_shapes
:?????????(2
flatten_7/Reshapes
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_8/Const?
flatten_8/ReshapeReshaperrow_conv2/Relu:activations:0flatten_8/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_8/Reshapes
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_9/Const?
flatten_9/ReshapeReshapercol_conv2/Relu:activations:0flatten_9/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_9/Reshapeu
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????<   2
flatten_10/Const?
flatten_10/ReshapeReshaperrow_conv/Relu:activations:0flatten_10/Const:output:0*
T0*'
_output_shapes
:?????????<2
flatten_10/Reshapeu
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????F   2
flatten_11/Const?
flatten_11/ReshapeReshapercol_conv/Relu:activations:0flatten_11/Const:output:0*
T0*'
_output_shapes
:?????????F2
flatten_11/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
flatten_1/Const?
flatten_1/ReshapeReshape ssquare_conv2/Relu:activations:0flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????(2
flatten_1/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_2/Const?
flatten_2/ReshapeReshapesrow_conv2/Relu:activations:0flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_2/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_3/Const?
flatten_3/ReshapeReshapescol_conv2/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_3/Reshapes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????<   2
flatten_4/Const?
flatten_4/ReshapeReshapesrow_conv/Relu:activations:0flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????<2
flatten_4/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????F   2
flatten_5/Const?
flatten_5/ReshapeReshapescol_conv/Relu:activations:0flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????F2
flatten_5/Reshapex
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2flatten_7/Reshape:output:0flatten_8/Reshape:output:0flatten_9/Reshape:output:0flatten_10/Reshape:output:0flatten_11/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_1/concatt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2flatten_1/Reshape:output:0flatten_2/Reshape:output:0flatten_3/Reshape:output:0flatten_4/Reshape:output:0flatten_5/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate/concat?
"rdense_layer/MatMul/ReadVariableOpReadVariableOp+rdense_layer_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02$
"rdense_layer/MatMul/ReadVariableOp?
rdense_layer/MatMulMatMulconcatenate_1/concat:output:0*rdense_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
rdense_layer/MatMul?
#rdense_layer/BiasAdd/ReadVariableOpReadVariableOp,rdense_layer_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#rdense_layer/BiasAdd/ReadVariableOp?
rdense_layer/BiasAddBiasAddrdense_layer/MatMul:product:0+rdense_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
rdense_layer/BiasAdd
rdense_layer/ReluRelurdense_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
rdense_layer/Relu?
"sdense_layer/MatMul/ReadVariableOpReadVariableOp+sdense_layer_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02$
"sdense_layer/MatMul/ReadVariableOp?
sdense_layer/MatMulMatMulconcatenate/concat:output:0*sdense_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sdense_layer/MatMul?
#sdense_layer/BiasAdd/ReadVariableOpReadVariableOp,sdense_layer_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#sdense_layer/BiasAdd/ReadVariableOp?
sdense_layer/BiasAddBiasAddsdense_layer/MatMul:product:0+sdense_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sdense_layer/BiasAdd
sdense_layer/ReluRelusdense_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sdense_layer/Relu?
#result_output/MatMul/ReadVariableOpReadVariableOp,result_output_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02%
#result_output/MatMul/ReadVariableOp?
result_output/MatMulMatMulrdense_layer/Relu:activations:0+result_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
result_output/MatMul?
$result_output/BiasAdd/ReadVariableOpReadVariableOp-result_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$result_output/BiasAdd/ReadVariableOp?
result_output/BiasAddBiasAddresult_output/MatMul:product:0,result_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
result_output/BiasAdd?
result_output/SigmoidSigmoidresult_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
result_output/Sigmoid?
"score_output/MatMul/ReadVariableOpReadVariableOp+score_output_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02$
"score_output/MatMul/ReadVariableOp?
score_output/MatMulMatMulsdense_layer/Relu:activations:0*score_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
score_output/MatMul?
#score_output/BiasAdd/ReadVariableOpReadVariableOp,score_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#score_output/BiasAdd/ReadVariableOp?
score_output/BiasAddBiasAddscore_output/MatMul:product:0+score_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
score_output/BiasAdd?
score_output/SoftmaxSoftmaxscore_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
score_output/Softmax?

IdentityIdentityscore_output/Softmax:softmax:0!^rcol_conv/BiasAdd/ReadVariableOp ^rcol_conv/Conv2D/ReadVariableOp"^rcol_conv2/BiasAdd/ReadVariableOp!^rcol_conv2/Conv2D/ReadVariableOp$^rdense_layer/BiasAdd/ReadVariableOp#^rdense_layer/MatMul/ReadVariableOp%^result_output/BiasAdd/ReadVariableOp$^result_output/MatMul/ReadVariableOp!^rrow_conv/BiasAdd/ReadVariableOp ^rrow_conv/Conv2D/ReadVariableOp"^rrow_conv2/BiasAdd/ReadVariableOp!^rrow_conv2/Conv2D/ReadVariableOp%^rsquare_conv1/BiasAdd/ReadVariableOp$^rsquare_conv1/Conv2D/ReadVariableOp%^rsquare_conv2/BiasAdd/ReadVariableOp$^rsquare_conv2/Conv2D/ReadVariableOp!^scol_conv/BiasAdd/ReadVariableOp ^scol_conv/Conv2D/ReadVariableOp"^scol_conv2/BiasAdd/ReadVariableOp!^scol_conv2/Conv2D/ReadVariableOp$^score_output/BiasAdd/ReadVariableOp#^score_output/MatMul/ReadVariableOp$^sdense_layer/BiasAdd/ReadVariableOp#^sdense_layer/MatMul/ReadVariableOp!^srow_conv/BiasAdd/ReadVariableOp ^srow_conv/Conv2D/ReadVariableOp"^srow_conv2/BiasAdd/ReadVariableOp!^srow_conv2/Conv2D/ReadVariableOp%^ssquare_conv1/BiasAdd/ReadVariableOp$^ssquare_conv1/Conv2D/ReadVariableOp%^ssquare_conv2/BiasAdd/ReadVariableOp$^ssquare_conv2/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?


Identity_1Identityresult_output/Sigmoid:y:0!^rcol_conv/BiasAdd/ReadVariableOp ^rcol_conv/Conv2D/ReadVariableOp"^rcol_conv2/BiasAdd/ReadVariableOp!^rcol_conv2/Conv2D/ReadVariableOp$^rdense_layer/BiasAdd/ReadVariableOp#^rdense_layer/MatMul/ReadVariableOp%^result_output/BiasAdd/ReadVariableOp$^result_output/MatMul/ReadVariableOp!^rrow_conv/BiasAdd/ReadVariableOp ^rrow_conv/Conv2D/ReadVariableOp"^rrow_conv2/BiasAdd/ReadVariableOp!^rrow_conv2/Conv2D/ReadVariableOp%^rsquare_conv1/BiasAdd/ReadVariableOp$^rsquare_conv1/Conv2D/ReadVariableOp%^rsquare_conv2/BiasAdd/ReadVariableOp$^rsquare_conv2/Conv2D/ReadVariableOp!^scol_conv/BiasAdd/ReadVariableOp ^scol_conv/Conv2D/ReadVariableOp"^scol_conv2/BiasAdd/ReadVariableOp!^scol_conv2/Conv2D/ReadVariableOp$^score_output/BiasAdd/ReadVariableOp#^score_output/MatMul/ReadVariableOp$^sdense_layer/BiasAdd/ReadVariableOp#^sdense_layer/MatMul/ReadVariableOp!^srow_conv/BiasAdd/ReadVariableOp ^srow_conv/Conv2D/ReadVariableOp"^srow_conv2/BiasAdd/ReadVariableOp!^srow_conv2/Conv2D/ReadVariableOp%^ssquare_conv1/BiasAdd/ReadVariableOp$^ssquare_conv1/Conv2D/ReadVariableOp%^ssquare_conv2/BiasAdd/ReadVariableOp$^ssquare_conv2/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 rcol_conv/BiasAdd/ReadVariableOp rcol_conv/BiasAdd/ReadVariableOp2B
rcol_conv/Conv2D/ReadVariableOprcol_conv/Conv2D/ReadVariableOp2F
!rcol_conv2/BiasAdd/ReadVariableOp!rcol_conv2/BiasAdd/ReadVariableOp2D
 rcol_conv2/Conv2D/ReadVariableOp rcol_conv2/Conv2D/ReadVariableOp2J
#rdense_layer/BiasAdd/ReadVariableOp#rdense_layer/BiasAdd/ReadVariableOp2H
"rdense_layer/MatMul/ReadVariableOp"rdense_layer/MatMul/ReadVariableOp2L
$result_output/BiasAdd/ReadVariableOp$result_output/BiasAdd/ReadVariableOp2J
#result_output/MatMul/ReadVariableOp#result_output/MatMul/ReadVariableOp2D
 rrow_conv/BiasAdd/ReadVariableOp rrow_conv/BiasAdd/ReadVariableOp2B
rrow_conv/Conv2D/ReadVariableOprrow_conv/Conv2D/ReadVariableOp2F
!rrow_conv2/BiasAdd/ReadVariableOp!rrow_conv2/BiasAdd/ReadVariableOp2D
 rrow_conv2/Conv2D/ReadVariableOp rrow_conv2/Conv2D/ReadVariableOp2L
$rsquare_conv1/BiasAdd/ReadVariableOp$rsquare_conv1/BiasAdd/ReadVariableOp2J
#rsquare_conv1/Conv2D/ReadVariableOp#rsquare_conv1/Conv2D/ReadVariableOp2L
$rsquare_conv2/BiasAdd/ReadVariableOp$rsquare_conv2/BiasAdd/ReadVariableOp2J
#rsquare_conv2/Conv2D/ReadVariableOp#rsquare_conv2/Conv2D/ReadVariableOp2D
 scol_conv/BiasAdd/ReadVariableOp scol_conv/BiasAdd/ReadVariableOp2B
scol_conv/Conv2D/ReadVariableOpscol_conv/Conv2D/ReadVariableOp2F
!scol_conv2/BiasAdd/ReadVariableOp!scol_conv2/BiasAdd/ReadVariableOp2D
 scol_conv2/Conv2D/ReadVariableOp scol_conv2/Conv2D/ReadVariableOp2J
#score_output/BiasAdd/ReadVariableOp#score_output/BiasAdd/ReadVariableOp2H
"score_output/MatMul/ReadVariableOp"score_output/MatMul/ReadVariableOp2J
#sdense_layer/BiasAdd/ReadVariableOp#sdense_layer/BiasAdd/ReadVariableOp2H
"sdense_layer/MatMul/ReadVariableOp"sdense_layer/MatMul/ReadVariableOp2D
 srow_conv/BiasAdd/ReadVariableOp srow_conv/BiasAdd/ReadVariableOp2B
srow_conv/Conv2D/ReadVariableOpsrow_conv/Conv2D/ReadVariableOp2F
!srow_conv2/BiasAdd/ReadVariableOp!srow_conv2/BiasAdd/ReadVariableOp2D
 srow_conv2/Conv2D/ReadVariableOp srow_conv2/Conv2D/ReadVariableOp2L
$ssquare_conv1/BiasAdd/ReadVariableOp$ssquare_conv1/BiasAdd/ReadVariableOp2J
#ssquare_conv1/Conv2D/ReadVariableOp#ssquare_conv1/Conv2D/ReadVariableOp2L
$ssquare_conv2/BiasAdd/ReadVariableOp$ssquare_conv2/BiasAdd/ReadVariableOp2J
#ssquare_conv2/Conv2D/ReadVariableOp#ssquare_conv2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
օ
?
@__inference_c4net_layer_call_and_return_conditional_losses_26718	
input-
rsquare_conv1_26624:!
rsquare_conv1_26626:-
ssquare_conv1_26629:!
ssquare_conv1_26631:)
rcol_conv_26634:

rcol_conv_26636:
)
rrow_conv_26639:

rrow_conv_26641:
*
rcol_conv2_26644:

rcol_conv2_26646:
*
rrow_conv2_26649:

rrow_conv2_26651:
-
rsquare_conv2_26654:!
rsquare_conv2_26656:)
scol_conv_26659:

scol_conv_26661:
)
srow_conv_26664:

srow_conv_26666:
*
scol_conv2_26669:

scol_conv2_26671:
*
srow_conv2_26674:

srow_conv2_26676:
-
ssquare_conv2_26679:!
ssquare_conv2_26681:%
rdense_layer_26696:	?d 
rdense_layer_26698:d%
sdense_layer_26701:	?d 
sdense_layer_26703:d%
result_output_26706:d!
result_output_26708:$
score_output_26711:d 
score_output_26713:
identity

identity_1??!rcol_conv/StatefulPartitionedCall?"rcol_conv2/StatefulPartitionedCall?$rdense_layer/StatefulPartitionedCall?%result_output/StatefulPartitionedCall?!rrow_conv/StatefulPartitionedCall?"rrow_conv2/StatefulPartitionedCall?%rsquare_conv1/StatefulPartitionedCall?%rsquare_conv2/StatefulPartitionedCall?!scol_conv/StatefulPartitionedCall?"scol_conv2/StatefulPartitionedCall?$score_output/StatefulPartitionedCall?$sdense_layer/StatefulPartitionedCall?!srow_conv/StatefulPartitionedCall?"srow_conv2/StatefulPartitionedCall?%ssquare_conv1/StatefulPartitionedCall?%ssquare_conv2/StatefulPartitionedCall?
%rsquare_conv1/StatefulPartitionedCallStatefulPartitionedCallinputrsquare_conv1_26624rsquare_conv1_26626*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rsquare_conv1_layer_call_and_return_conditional_losses_255382'
%rsquare_conv1/StatefulPartitionedCall?
%ssquare_conv1/StatefulPartitionedCallStatefulPartitionedCallinputssquare_conv1_26629ssquare_conv1_26631*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_ssquare_conv1_layer_call_and_return_conditional_losses_255552'
%ssquare_conv1/StatefulPartitionedCall?
!rcol_conv/StatefulPartitionedCallStatefulPartitionedCallinputrcol_conv_26634rcol_conv_26636*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_rcol_conv_layer_call_and_return_conditional_losses_255722#
!rcol_conv/StatefulPartitionedCall?
!rrow_conv/StatefulPartitionedCallStatefulPartitionedCallinputrrow_conv_26639rrow_conv_26641*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_rrow_conv_layer_call_and_return_conditional_losses_255892#
!rrow_conv/StatefulPartitionedCall?
"rcol_conv2/StatefulPartitionedCallStatefulPartitionedCall.rsquare_conv1/StatefulPartitionedCall:output:0rcol_conv2_26644rcol_conv2_26646*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_rcol_conv2_layer_call_and_return_conditional_losses_256062$
"rcol_conv2/StatefulPartitionedCall?
"rrow_conv2/StatefulPartitionedCallStatefulPartitionedCall.rsquare_conv1/StatefulPartitionedCall:output:0rrow_conv2_26649rrow_conv2_26651*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_rrow_conv2_layer_call_and_return_conditional_losses_256232$
"rrow_conv2/StatefulPartitionedCall?
%rsquare_conv2/StatefulPartitionedCallStatefulPartitionedCall.rsquare_conv1/StatefulPartitionedCall:output:0rsquare_conv2_26654rsquare_conv2_26656*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rsquare_conv2_layer_call_and_return_conditional_losses_256402'
%rsquare_conv2/StatefulPartitionedCall?
!scol_conv/StatefulPartitionedCallStatefulPartitionedCallinputscol_conv_26659scol_conv_26661*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_scol_conv_layer_call_and_return_conditional_losses_256572#
!scol_conv/StatefulPartitionedCall?
!srow_conv/StatefulPartitionedCallStatefulPartitionedCallinputsrow_conv_26664srow_conv_26666*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_srow_conv_layer_call_and_return_conditional_losses_256742#
!srow_conv/StatefulPartitionedCall?
"scol_conv2/StatefulPartitionedCallStatefulPartitionedCall.ssquare_conv1/StatefulPartitionedCall:output:0scol_conv2_26669scol_conv2_26671*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_scol_conv2_layer_call_and_return_conditional_losses_256912$
"scol_conv2/StatefulPartitionedCall?
"srow_conv2/StatefulPartitionedCallStatefulPartitionedCall.ssquare_conv1/StatefulPartitionedCall:output:0srow_conv2_26674srow_conv2_26676*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_srow_conv2_layer_call_and_return_conditional_losses_257082$
"srow_conv2/StatefulPartitionedCall?
%ssquare_conv2/StatefulPartitionedCallStatefulPartitionedCall.ssquare_conv1/StatefulPartitionedCall:output:0ssquare_conv2_26679ssquare_conv2_26681*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_ssquare_conv2_layer_call_and_return_conditional_losses_257252'
%ssquare_conv2/StatefulPartitionedCall?
flatten_7/PartitionedCallPartitionedCall.rsquare_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_257372
flatten_7/PartitionedCall?
flatten_8/PartitionedCallPartitionedCall+rrow_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_257452
flatten_8/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall+rcol_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_9_layer_call_and_return_conditional_losses_257532
flatten_9/PartitionedCall?
flatten_10/PartitionedCallPartitionedCall*rrow_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_10_layer_call_and_return_conditional_losses_257612
flatten_10/PartitionedCall?
flatten_11/PartitionedCallPartitionedCall*rcol_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_11_layer_call_and_return_conditional_losses_257692
flatten_11/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall.ssquare_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_257772
flatten_1/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall+srow_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_257852
flatten_2/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall+scol_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_257932
flatten_3/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall*srow_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_258012
flatten_4/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall*scol_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_258092
flatten_5/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall"flatten_7/PartitionedCall:output:0"flatten_8/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0#flatten_10/PartitionedCall:output:0#flatten_11/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_258212
concatenate_1/PartitionedCall?
concatenate/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_258332
concatenate/PartitionedCall?
$rdense_layer/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0rdense_layer_26696rdense_layer_26698*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_rdense_layer_layer_call_and_return_conditional_losses_258462&
$rdense_layer/StatefulPartitionedCall?
$sdense_layer/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sdense_layer_26701sdense_layer_26703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sdense_layer_layer_call_and_return_conditional_losses_258632&
$sdense_layer/StatefulPartitionedCall?
%result_output/StatefulPartitionedCallStatefulPartitionedCall-rdense_layer/StatefulPartitionedCall:output:0result_output_26706result_output_26708*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_result_output_layer_call_and_return_conditional_losses_258802'
%result_output/StatefulPartitionedCall?
$score_output/StatefulPartitionedCallStatefulPartitionedCall-sdense_layer/StatefulPartitionedCall:output:0score_output_26711score_output_26713*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_score_output_layer_call_and_return_conditional_losses_258972&
$score_output/StatefulPartitionedCall?
IdentityIdentity-score_output/StatefulPartitionedCall:output:0"^rcol_conv/StatefulPartitionedCall#^rcol_conv2/StatefulPartitionedCall%^rdense_layer/StatefulPartitionedCall&^result_output/StatefulPartitionedCall"^rrow_conv/StatefulPartitionedCall#^rrow_conv2/StatefulPartitionedCall&^rsquare_conv1/StatefulPartitionedCall&^rsquare_conv2/StatefulPartitionedCall"^scol_conv/StatefulPartitionedCall#^scol_conv2/StatefulPartitionedCall%^score_output/StatefulPartitionedCall%^sdense_layer/StatefulPartitionedCall"^srow_conv/StatefulPartitionedCall#^srow_conv2/StatefulPartitionedCall&^ssquare_conv1/StatefulPartitionedCall&^ssquare_conv2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity.result_output/StatefulPartitionedCall:output:0"^rcol_conv/StatefulPartitionedCall#^rcol_conv2/StatefulPartitionedCall%^rdense_layer/StatefulPartitionedCall&^result_output/StatefulPartitionedCall"^rrow_conv/StatefulPartitionedCall#^rrow_conv2/StatefulPartitionedCall&^rsquare_conv1/StatefulPartitionedCall&^rsquare_conv2/StatefulPartitionedCall"^scol_conv/StatefulPartitionedCall#^scol_conv2/StatefulPartitionedCall%^score_output/StatefulPartitionedCall%^sdense_layer/StatefulPartitionedCall"^srow_conv/StatefulPartitionedCall#^srow_conv2/StatefulPartitionedCall&^ssquare_conv1/StatefulPartitionedCall&^ssquare_conv2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!rcol_conv/StatefulPartitionedCall!rcol_conv/StatefulPartitionedCall2H
"rcol_conv2/StatefulPartitionedCall"rcol_conv2/StatefulPartitionedCall2L
$rdense_layer/StatefulPartitionedCall$rdense_layer/StatefulPartitionedCall2N
%result_output/StatefulPartitionedCall%result_output/StatefulPartitionedCall2F
!rrow_conv/StatefulPartitionedCall!rrow_conv/StatefulPartitionedCall2H
"rrow_conv2/StatefulPartitionedCall"rrow_conv2/StatefulPartitionedCall2N
%rsquare_conv1/StatefulPartitionedCall%rsquare_conv1/StatefulPartitionedCall2N
%rsquare_conv2/StatefulPartitionedCall%rsquare_conv2/StatefulPartitionedCall2F
!scol_conv/StatefulPartitionedCall!scol_conv/StatefulPartitionedCall2H
"scol_conv2/StatefulPartitionedCall"scol_conv2/StatefulPartitionedCall2L
$score_output/StatefulPartitionedCall$score_output/StatefulPartitionedCall2L
$sdense_layer/StatefulPartitionedCall$sdense_layer/StatefulPartitionedCall2F
!srow_conv/StatefulPartitionedCall!srow_conv/StatefulPartitionedCall2H
"srow_conv2/StatefulPartitionedCall"srow_conv2/StatefulPartitionedCall2N
%ssquare_conv1/StatefulPartitionedCall%ssquare_conv1/StatefulPartitionedCall2N
%ssquare_conv2/StatefulPartitionedCall%ssquare_conv2/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
-__inference_rsquare_conv1_layer_call_fn_27255

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rsquare_conv1_layer_call_and_return_conditional_losses_255382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_rrow_conv2_layer_call_and_return_conditional_losses_25623

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_25777

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????(2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_srow_conv2_layer_call_and_return_conditional_losses_27286

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_rrow_conv2_layer_call_fn_27395

inputs!
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_rrow_conv2_layer_call_and_return_conditional_losses_256232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_srow_conv_layer_call_fn_27335

inputs!
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_srow_conv_layer_call_and_return_conditional_losses_256742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_27472

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_25821

inputs
inputs_1
inputs_2
inputs_3
inputs_4
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????(:?????????:?????????:?????????<:?????????F:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????<
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
-__inference_result_output_layer_call_fn_27683

inputs
unknown:d
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_result_output_layer_call_and_return_conditional_losses_258802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
%__inference_c4net_layer_call_fn_27144

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:

	unknown_4:
#
	unknown_5:

	unknown_6:
#
	unknown_7:

	unknown_8:
#
	unknown_9:


unknown_10:
$

unknown_11:

unknown_12:$

unknown_13:


unknown_14:
$

unknown_15:


unknown_16:
$

unknown_17:


unknown_18:
$

unknown_19:


unknown_20:
$

unknown_21:

unknown_22:

unknown_23:	?d

unknown_24:d

unknown_25:	?d

unknown_26:d

unknown_27:d

unknown_28:

unknown_29:d

unknown_30:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_c4net_layer_call_and_return_conditional_losses_259052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_10_layer_call_and_return_conditional_losses_25761

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????<   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????<2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
E
)__inference_flatten_9_layer_call_fn_27543

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_9_layer_call_and_return_conditional_losses_257532
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
%__inference_c4net_layer_call_fn_26524	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:

	unknown_4:
#
	unknown_5:

	unknown_6:
#
	unknown_7:

	unknown_8:
#
	unknown_9:


unknown_10:
$

unknown_11:

unknown_12:$

unknown_13:


unknown_14:
$

unknown_15:


unknown_16:
$

unknown_17:


unknown_18:
$

unknown_19:


unknown_20:
$

unknown_21:

unknown_22:

unknown_23:	?d

unknown_24:d

unknown_25:	?d

unknown_26:d

unknown_27:d

unknown_28:

unknown_29:d

unknown_30:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_c4net_layer_call_and_return_conditional_losses_263842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
D__inference_rcol_conv_layer_call_and_return_conditional_losses_27446

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_11_layer_call_and_return_conditional_losses_27560

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????F   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????F2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?

?
G__inference_score_output_layer_call_and_return_conditional_losses_27654

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
`
D__inference_flatten_8_layer_call_and_return_conditional_losses_27527

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
`
D__inference_flatten_8_layer_call_and_return_conditional_losses_25745

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
E__inference_rcol_conv2_layer_call_and_return_conditional_losses_27406

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_rrow_conv_layer_call_and_return_conditional_losses_27426

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sdense_layer_layer_call_fn_27623

inputs
unknown:	?d
	unknown_0:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sdense_layer_layer_call_and_return_conditional_losses_258632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_ssquare_conv2_layer_call_and_return_conditional_losses_25725

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
@__inference_c4net_layer_call_and_return_conditional_losses_27073

inputsF
,rsquare_conv1_conv2d_readvariableop_resource:;
-rsquare_conv1_biasadd_readvariableop_resource:F
,ssquare_conv1_conv2d_readvariableop_resource:;
-ssquare_conv1_biasadd_readvariableop_resource:B
(rcol_conv_conv2d_readvariableop_resource:
7
)rcol_conv_biasadd_readvariableop_resource:
B
(rrow_conv_conv2d_readvariableop_resource:
7
)rrow_conv_biasadd_readvariableop_resource:
C
)rcol_conv2_conv2d_readvariableop_resource:
8
*rcol_conv2_biasadd_readvariableop_resource:
C
)rrow_conv2_conv2d_readvariableop_resource:
8
*rrow_conv2_biasadd_readvariableop_resource:
F
,rsquare_conv2_conv2d_readvariableop_resource:;
-rsquare_conv2_biasadd_readvariableop_resource:B
(scol_conv_conv2d_readvariableop_resource:
7
)scol_conv_biasadd_readvariableop_resource:
B
(srow_conv_conv2d_readvariableop_resource:
7
)srow_conv_biasadd_readvariableop_resource:
C
)scol_conv2_conv2d_readvariableop_resource:
8
*scol_conv2_biasadd_readvariableop_resource:
C
)srow_conv2_conv2d_readvariableop_resource:
8
*srow_conv2_biasadd_readvariableop_resource:
F
,ssquare_conv2_conv2d_readvariableop_resource:;
-ssquare_conv2_biasadd_readvariableop_resource:>
+rdense_layer_matmul_readvariableop_resource:	?d:
,rdense_layer_biasadd_readvariableop_resource:d>
+sdense_layer_matmul_readvariableop_resource:	?d:
,sdense_layer_biasadd_readvariableop_resource:d>
,result_output_matmul_readvariableop_resource:d;
-result_output_biasadd_readvariableop_resource:=
+score_output_matmul_readvariableop_resource:d:
,score_output_biasadd_readvariableop_resource:
identity

identity_1?? rcol_conv/BiasAdd/ReadVariableOp?rcol_conv/Conv2D/ReadVariableOp?!rcol_conv2/BiasAdd/ReadVariableOp? rcol_conv2/Conv2D/ReadVariableOp?#rdense_layer/BiasAdd/ReadVariableOp?"rdense_layer/MatMul/ReadVariableOp?$result_output/BiasAdd/ReadVariableOp?#result_output/MatMul/ReadVariableOp? rrow_conv/BiasAdd/ReadVariableOp?rrow_conv/Conv2D/ReadVariableOp?!rrow_conv2/BiasAdd/ReadVariableOp? rrow_conv2/Conv2D/ReadVariableOp?$rsquare_conv1/BiasAdd/ReadVariableOp?#rsquare_conv1/Conv2D/ReadVariableOp?$rsquare_conv2/BiasAdd/ReadVariableOp?#rsquare_conv2/Conv2D/ReadVariableOp? scol_conv/BiasAdd/ReadVariableOp?scol_conv/Conv2D/ReadVariableOp?!scol_conv2/BiasAdd/ReadVariableOp? scol_conv2/Conv2D/ReadVariableOp?#score_output/BiasAdd/ReadVariableOp?"score_output/MatMul/ReadVariableOp?#sdense_layer/BiasAdd/ReadVariableOp?"sdense_layer/MatMul/ReadVariableOp? srow_conv/BiasAdd/ReadVariableOp?srow_conv/Conv2D/ReadVariableOp?!srow_conv2/BiasAdd/ReadVariableOp? srow_conv2/Conv2D/ReadVariableOp?$ssquare_conv1/BiasAdd/ReadVariableOp?#ssquare_conv1/Conv2D/ReadVariableOp?$ssquare_conv2/BiasAdd/ReadVariableOp?#ssquare_conv2/Conv2D/ReadVariableOp?
#rsquare_conv1/Conv2D/ReadVariableOpReadVariableOp,rsquare_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#rsquare_conv1/Conv2D/ReadVariableOp?
rsquare_conv1/Conv2DConv2Dinputs+rsquare_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
rsquare_conv1/Conv2D?
$rsquare_conv1/BiasAdd/ReadVariableOpReadVariableOp-rsquare_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$rsquare_conv1/BiasAdd/ReadVariableOp?
rsquare_conv1/BiasAddBiasAddrsquare_conv1/Conv2D:output:0,rsquare_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
rsquare_conv1/BiasAdd?
rsquare_conv1/ReluRelursquare_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
rsquare_conv1/Relu?
#ssquare_conv1/Conv2D/ReadVariableOpReadVariableOp,ssquare_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#ssquare_conv1/Conv2D/ReadVariableOp?
ssquare_conv1/Conv2DConv2Dinputs+ssquare_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ssquare_conv1/Conv2D?
$ssquare_conv1/BiasAdd/ReadVariableOpReadVariableOp-ssquare_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ssquare_conv1/BiasAdd/ReadVariableOp?
ssquare_conv1/BiasAddBiasAddssquare_conv1/Conv2D:output:0,ssquare_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ssquare_conv1/BiasAdd?
ssquare_conv1/ReluRelussquare_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
ssquare_conv1/Relu?
rcol_conv/Conv2D/ReadVariableOpReadVariableOp(rcol_conv_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02!
rcol_conv/Conv2D/ReadVariableOp?
rcol_conv/Conv2DConv2Dinputs'rcol_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
rcol_conv/Conv2D?
 rcol_conv/BiasAdd/ReadVariableOpReadVariableOp)rcol_conv_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 rcol_conv/BiasAdd/ReadVariableOp?
rcol_conv/BiasAddBiasAddrcol_conv/Conv2D:output:0(rcol_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
rcol_conv/BiasAdd~
rcol_conv/ReluRelurcol_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
rcol_conv/Relu?
rrow_conv/Conv2D/ReadVariableOpReadVariableOp(rrow_conv_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02!
rrow_conv/Conv2D/ReadVariableOp?
rrow_conv/Conv2DConv2Dinputs'rrow_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
rrow_conv/Conv2D?
 rrow_conv/BiasAdd/ReadVariableOpReadVariableOp)rrow_conv_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 rrow_conv/BiasAdd/ReadVariableOp?
rrow_conv/BiasAddBiasAddrrow_conv/Conv2D:output:0(rrow_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
rrow_conv/BiasAdd~
rrow_conv/ReluRelurrow_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
rrow_conv/Relu?
 rcol_conv2/Conv2D/ReadVariableOpReadVariableOp)rcol_conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02"
 rcol_conv2/Conv2D/ReadVariableOp?
rcol_conv2/Conv2DConv2D rsquare_conv1/Relu:activations:0(rcol_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
rcol_conv2/Conv2D?
!rcol_conv2/BiasAdd/ReadVariableOpReadVariableOp*rcol_conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02#
!rcol_conv2/BiasAdd/ReadVariableOp?
rcol_conv2/BiasAddBiasAddrcol_conv2/Conv2D:output:0)rcol_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
rcol_conv2/BiasAdd?
rcol_conv2/ReluRelurcol_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
rcol_conv2/Relu?
 rrow_conv2/Conv2D/ReadVariableOpReadVariableOp)rrow_conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02"
 rrow_conv2/Conv2D/ReadVariableOp?
rrow_conv2/Conv2DConv2D rsquare_conv1/Relu:activations:0(rrow_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
rrow_conv2/Conv2D?
!rrow_conv2/BiasAdd/ReadVariableOpReadVariableOp*rrow_conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02#
!rrow_conv2/BiasAdd/ReadVariableOp?
rrow_conv2/BiasAddBiasAddrrow_conv2/Conv2D:output:0)rrow_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
rrow_conv2/BiasAdd?
rrow_conv2/ReluRelurrow_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
rrow_conv2/Relu?
#rsquare_conv2/Conv2D/ReadVariableOpReadVariableOp,rsquare_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#rsquare_conv2/Conv2D/ReadVariableOp?
rsquare_conv2/Conv2DConv2D rsquare_conv1/Relu:activations:0+rsquare_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
rsquare_conv2/Conv2D?
$rsquare_conv2/BiasAdd/ReadVariableOpReadVariableOp-rsquare_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$rsquare_conv2/BiasAdd/ReadVariableOp?
rsquare_conv2/BiasAddBiasAddrsquare_conv2/Conv2D:output:0,rsquare_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
rsquare_conv2/BiasAdd?
rsquare_conv2/ReluRelursquare_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
rsquare_conv2/Relu?
scol_conv/Conv2D/ReadVariableOpReadVariableOp(scol_conv_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02!
scol_conv/Conv2D/ReadVariableOp?
scol_conv/Conv2DConv2Dinputs'scol_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
scol_conv/Conv2D?
 scol_conv/BiasAdd/ReadVariableOpReadVariableOp)scol_conv_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 scol_conv/BiasAdd/ReadVariableOp?
scol_conv/BiasAddBiasAddscol_conv/Conv2D:output:0(scol_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
scol_conv/BiasAdd~
scol_conv/ReluReluscol_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
scol_conv/Relu?
srow_conv/Conv2D/ReadVariableOpReadVariableOp(srow_conv_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02!
srow_conv/Conv2D/ReadVariableOp?
srow_conv/Conv2DConv2Dinputs'srow_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
srow_conv/Conv2D?
 srow_conv/BiasAdd/ReadVariableOpReadVariableOp)srow_conv_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 srow_conv/BiasAdd/ReadVariableOp?
srow_conv/BiasAddBiasAddsrow_conv/Conv2D:output:0(srow_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
srow_conv/BiasAdd~
srow_conv/ReluRelusrow_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
srow_conv/Relu?
 scol_conv2/Conv2D/ReadVariableOpReadVariableOp)scol_conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02"
 scol_conv2/Conv2D/ReadVariableOp?
scol_conv2/Conv2DConv2D ssquare_conv1/Relu:activations:0(scol_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
scol_conv2/Conv2D?
!scol_conv2/BiasAdd/ReadVariableOpReadVariableOp*scol_conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02#
!scol_conv2/BiasAdd/ReadVariableOp?
scol_conv2/BiasAddBiasAddscol_conv2/Conv2D:output:0)scol_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
scol_conv2/BiasAdd?
scol_conv2/ReluReluscol_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
scol_conv2/Relu?
 srow_conv2/Conv2D/ReadVariableOpReadVariableOp)srow_conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02"
 srow_conv2/Conv2D/ReadVariableOp?
srow_conv2/Conv2DConv2D ssquare_conv1/Relu:activations:0(srow_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
srow_conv2/Conv2D?
!srow_conv2/BiasAdd/ReadVariableOpReadVariableOp*srow_conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02#
!srow_conv2/BiasAdd/ReadVariableOp?
srow_conv2/BiasAddBiasAddsrow_conv2/Conv2D:output:0)srow_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
srow_conv2/BiasAdd?
srow_conv2/ReluRelusrow_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
srow_conv2/Relu?
#ssquare_conv2/Conv2D/ReadVariableOpReadVariableOp,ssquare_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#ssquare_conv2/Conv2D/ReadVariableOp?
ssquare_conv2/Conv2DConv2D ssquare_conv1/Relu:activations:0+ssquare_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ssquare_conv2/Conv2D?
$ssquare_conv2/BiasAdd/ReadVariableOpReadVariableOp-ssquare_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ssquare_conv2/BiasAdd/ReadVariableOp?
ssquare_conv2/BiasAddBiasAddssquare_conv2/Conv2D:output:0,ssquare_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ssquare_conv2/BiasAdd?
ssquare_conv2/ReluRelussquare_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
ssquare_conv2/Relus
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
flatten_7/Const?
flatten_7/ReshapeReshape rsquare_conv2/Relu:activations:0flatten_7/Const:output:0*
T0*'
_output_shapes
:?????????(2
flatten_7/Reshapes
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_8/Const?
flatten_8/ReshapeReshaperrow_conv2/Relu:activations:0flatten_8/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_8/Reshapes
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_9/Const?
flatten_9/ReshapeReshapercol_conv2/Relu:activations:0flatten_9/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_9/Reshapeu
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????<   2
flatten_10/Const?
flatten_10/ReshapeReshaperrow_conv/Relu:activations:0flatten_10/Const:output:0*
T0*'
_output_shapes
:?????????<2
flatten_10/Reshapeu
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????F   2
flatten_11/Const?
flatten_11/ReshapeReshapercol_conv/Relu:activations:0flatten_11/Const:output:0*
T0*'
_output_shapes
:?????????F2
flatten_11/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
flatten_1/Const?
flatten_1/ReshapeReshape ssquare_conv2/Relu:activations:0flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????(2
flatten_1/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_2/Const?
flatten_2/ReshapeReshapesrow_conv2/Relu:activations:0flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_2/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_3/Const?
flatten_3/ReshapeReshapescol_conv2/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_3/Reshapes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????<   2
flatten_4/Const?
flatten_4/ReshapeReshapesrow_conv/Relu:activations:0flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????<2
flatten_4/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????F   2
flatten_5/Const?
flatten_5/ReshapeReshapescol_conv/Relu:activations:0flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????F2
flatten_5/Reshapex
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2flatten_7/Reshape:output:0flatten_8/Reshape:output:0flatten_9/Reshape:output:0flatten_10/Reshape:output:0flatten_11/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_1/concatt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2flatten_1/Reshape:output:0flatten_2/Reshape:output:0flatten_3/Reshape:output:0flatten_4/Reshape:output:0flatten_5/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate/concat?
"rdense_layer/MatMul/ReadVariableOpReadVariableOp+rdense_layer_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02$
"rdense_layer/MatMul/ReadVariableOp?
rdense_layer/MatMulMatMulconcatenate_1/concat:output:0*rdense_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
rdense_layer/MatMul?
#rdense_layer/BiasAdd/ReadVariableOpReadVariableOp,rdense_layer_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#rdense_layer/BiasAdd/ReadVariableOp?
rdense_layer/BiasAddBiasAddrdense_layer/MatMul:product:0+rdense_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
rdense_layer/BiasAdd
rdense_layer/ReluRelurdense_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
rdense_layer/Relu?
"sdense_layer/MatMul/ReadVariableOpReadVariableOp+sdense_layer_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02$
"sdense_layer/MatMul/ReadVariableOp?
sdense_layer/MatMulMatMulconcatenate/concat:output:0*sdense_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sdense_layer/MatMul?
#sdense_layer/BiasAdd/ReadVariableOpReadVariableOp,sdense_layer_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#sdense_layer/BiasAdd/ReadVariableOp?
sdense_layer/BiasAddBiasAddsdense_layer/MatMul:product:0+sdense_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sdense_layer/BiasAdd
sdense_layer/ReluRelusdense_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sdense_layer/Relu?
#result_output/MatMul/ReadVariableOpReadVariableOp,result_output_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02%
#result_output/MatMul/ReadVariableOp?
result_output/MatMulMatMulrdense_layer/Relu:activations:0+result_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
result_output/MatMul?
$result_output/BiasAdd/ReadVariableOpReadVariableOp-result_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$result_output/BiasAdd/ReadVariableOp?
result_output/BiasAddBiasAddresult_output/MatMul:product:0,result_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
result_output/BiasAdd?
result_output/SigmoidSigmoidresult_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
result_output/Sigmoid?
"score_output/MatMul/ReadVariableOpReadVariableOp+score_output_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02$
"score_output/MatMul/ReadVariableOp?
score_output/MatMulMatMulsdense_layer/Relu:activations:0*score_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
score_output/MatMul?
#score_output/BiasAdd/ReadVariableOpReadVariableOp,score_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#score_output/BiasAdd/ReadVariableOp?
score_output/BiasAddBiasAddscore_output/MatMul:product:0+score_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
score_output/BiasAdd?
score_output/SoftmaxSoftmaxscore_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
score_output/Softmax?

IdentityIdentityscore_output/Softmax:softmax:0!^rcol_conv/BiasAdd/ReadVariableOp ^rcol_conv/Conv2D/ReadVariableOp"^rcol_conv2/BiasAdd/ReadVariableOp!^rcol_conv2/Conv2D/ReadVariableOp$^rdense_layer/BiasAdd/ReadVariableOp#^rdense_layer/MatMul/ReadVariableOp%^result_output/BiasAdd/ReadVariableOp$^result_output/MatMul/ReadVariableOp!^rrow_conv/BiasAdd/ReadVariableOp ^rrow_conv/Conv2D/ReadVariableOp"^rrow_conv2/BiasAdd/ReadVariableOp!^rrow_conv2/Conv2D/ReadVariableOp%^rsquare_conv1/BiasAdd/ReadVariableOp$^rsquare_conv1/Conv2D/ReadVariableOp%^rsquare_conv2/BiasAdd/ReadVariableOp$^rsquare_conv2/Conv2D/ReadVariableOp!^scol_conv/BiasAdd/ReadVariableOp ^scol_conv/Conv2D/ReadVariableOp"^scol_conv2/BiasAdd/ReadVariableOp!^scol_conv2/Conv2D/ReadVariableOp$^score_output/BiasAdd/ReadVariableOp#^score_output/MatMul/ReadVariableOp$^sdense_layer/BiasAdd/ReadVariableOp#^sdense_layer/MatMul/ReadVariableOp!^srow_conv/BiasAdd/ReadVariableOp ^srow_conv/Conv2D/ReadVariableOp"^srow_conv2/BiasAdd/ReadVariableOp!^srow_conv2/Conv2D/ReadVariableOp%^ssquare_conv1/BiasAdd/ReadVariableOp$^ssquare_conv1/Conv2D/ReadVariableOp%^ssquare_conv2/BiasAdd/ReadVariableOp$^ssquare_conv2/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?


Identity_1Identityresult_output/Sigmoid:y:0!^rcol_conv/BiasAdd/ReadVariableOp ^rcol_conv/Conv2D/ReadVariableOp"^rcol_conv2/BiasAdd/ReadVariableOp!^rcol_conv2/Conv2D/ReadVariableOp$^rdense_layer/BiasAdd/ReadVariableOp#^rdense_layer/MatMul/ReadVariableOp%^result_output/BiasAdd/ReadVariableOp$^result_output/MatMul/ReadVariableOp!^rrow_conv/BiasAdd/ReadVariableOp ^rrow_conv/Conv2D/ReadVariableOp"^rrow_conv2/BiasAdd/ReadVariableOp!^rrow_conv2/Conv2D/ReadVariableOp%^rsquare_conv1/BiasAdd/ReadVariableOp$^rsquare_conv1/Conv2D/ReadVariableOp%^rsquare_conv2/BiasAdd/ReadVariableOp$^rsquare_conv2/Conv2D/ReadVariableOp!^scol_conv/BiasAdd/ReadVariableOp ^scol_conv/Conv2D/ReadVariableOp"^scol_conv2/BiasAdd/ReadVariableOp!^scol_conv2/Conv2D/ReadVariableOp$^score_output/BiasAdd/ReadVariableOp#^score_output/MatMul/ReadVariableOp$^sdense_layer/BiasAdd/ReadVariableOp#^sdense_layer/MatMul/ReadVariableOp!^srow_conv/BiasAdd/ReadVariableOp ^srow_conv/Conv2D/ReadVariableOp"^srow_conv2/BiasAdd/ReadVariableOp!^srow_conv2/Conv2D/ReadVariableOp%^ssquare_conv1/BiasAdd/ReadVariableOp$^ssquare_conv1/Conv2D/ReadVariableOp%^ssquare_conv2/BiasAdd/ReadVariableOp$^ssquare_conv2/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 rcol_conv/BiasAdd/ReadVariableOp rcol_conv/BiasAdd/ReadVariableOp2B
rcol_conv/Conv2D/ReadVariableOprcol_conv/Conv2D/ReadVariableOp2F
!rcol_conv2/BiasAdd/ReadVariableOp!rcol_conv2/BiasAdd/ReadVariableOp2D
 rcol_conv2/Conv2D/ReadVariableOp rcol_conv2/Conv2D/ReadVariableOp2J
#rdense_layer/BiasAdd/ReadVariableOp#rdense_layer/BiasAdd/ReadVariableOp2H
"rdense_layer/MatMul/ReadVariableOp"rdense_layer/MatMul/ReadVariableOp2L
$result_output/BiasAdd/ReadVariableOp$result_output/BiasAdd/ReadVariableOp2J
#result_output/MatMul/ReadVariableOp#result_output/MatMul/ReadVariableOp2D
 rrow_conv/BiasAdd/ReadVariableOp rrow_conv/BiasAdd/ReadVariableOp2B
rrow_conv/Conv2D/ReadVariableOprrow_conv/Conv2D/ReadVariableOp2F
!rrow_conv2/BiasAdd/ReadVariableOp!rrow_conv2/BiasAdd/ReadVariableOp2D
 rrow_conv2/Conv2D/ReadVariableOp rrow_conv2/Conv2D/ReadVariableOp2L
$rsquare_conv1/BiasAdd/ReadVariableOp$rsquare_conv1/BiasAdd/ReadVariableOp2J
#rsquare_conv1/Conv2D/ReadVariableOp#rsquare_conv1/Conv2D/ReadVariableOp2L
$rsquare_conv2/BiasAdd/ReadVariableOp$rsquare_conv2/BiasAdd/ReadVariableOp2J
#rsquare_conv2/Conv2D/ReadVariableOp#rsquare_conv2/Conv2D/ReadVariableOp2D
 scol_conv/BiasAdd/ReadVariableOp scol_conv/BiasAdd/ReadVariableOp2B
scol_conv/Conv2D/ReadVariableOpscol_conv/Conv2D/ReadVariableOp2F
!scol_conv2/BiasAdd/ReadVariableOp!scol_conv2/BiasAdd/ReadVariableOp2D
 scol_conv2/Conv2D/ReadVariableOp scol_conv2/Conv2D/ReadVariableOp2J
#score_output/BiasAdd/ReadVariableOp#score_output/BiasAdd/ReadVariableOp2H
"score_output/MatMul/ReadVariableOp"score_output/MatMul/ReadVariableOp2J
#sdense_layer/BiasAdd/ReadVariableOp#sdense_layer/BiasAdd/ReadVariableOp2H
"sdense_layer/MatMul/ReadVariableOp"sdense_layer/MatMul/ReadVariableOp2D
 srow_conv/BiasAdd/ReadVariableOp srow_conv/BiasAdd/ReadVariableOp2B
srow_conv/Conv2D/ReadVariableOpsrow_conv/Conv2D/ReadVariableOp2F
!srow_conv2/BiasAdd/ReadVariableOp!srow_conv2/BiasAdd/ReadVariableOp2D
 srow_conv2/Conv2D/ReadVariableOp srow_conv2/Conv2D/ReadVariableOp2L
$ssquare_conv1/BiasAdd/ReadVariableOp$ssquare_conv1/BiasAdd/ReadVariableOp2J
#ssquare_conv1/Conv2D/ReadVariableOp#ssquare_conv1/Conv2D/ReadVariableOp2L
$ssquare_conv2/BiasAdd/ReadVariableOp$ssquare_conv2/BiasAdd/ReadVariableOp2J
#ssquare_conv2/Conv2D/ReadVariableOp#ssquare_conv2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_9_layer_call_and_return_conditional_losses_27538

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
E
)__inference_flatten_2_layer_call_fn_27477

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_257852
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
-__inference_ssquare_conv2_layer_call_fn_27275

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_ssquare_conv2_layer_call_and_return_conditional_losses_257252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_7_layer_call_fn_27521

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_257372
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_9_layer_call_and_return_conditional_losses_25753

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_25793

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
F__inference_concatenate_layer_call_and_return_conditional_losses_27575
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????(:?????????:?????????:?????????<:?????????F:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????<
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????F
"
_user_specified_name
inputs/4
?
?
,__inference_score_output_layer_call_fn_27663

inputs
unknown:d
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_score_output_layer_call_and_return_conditional_losses_258972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
F
*__inference_flatten_11_layer_call_fn_27565

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_11_layer_call_and_return_conditional_losses_257692
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
a
E__inference_flatten_11_layer_call_and_return_conditional_losses_25769

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????F   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????F2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_26791	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:

	unknown_4:
#
	unknown_5:

	unknown_6:
#
	unknown_7:

	unknown_8:
#
	unknown_9:


unknown_10:
$

unknown_11:

unknown_12:$

unknown_13:


unknown_14:
$

unknown_15:


unknown_16:
$

unknown_17:


unknown_18:
$

unknown_19:


unknown_20:
$

unknown_21:

unknown_22:

unknown_23:	?d

unknown_24:d

unknown_25:	?d

unknown_26:d

unknown_27:d

unknown_28:

unknown_29:d

unknown_30:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_255202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
)__inference_rcol_conv_layer_call_fn_27455

inputs!
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_rcol_conv_layer_call_and_return_conditional_losses_255722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_1_layer_call_fn_27466

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_257772
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_27494

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????<   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????<2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
`
D__inference_flatten_7_layer_call_and_return_conditional_losses_27516

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????(2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_srow_conv2_layer_call_and_return_conditional_losses_25708

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_scol_conv_layer_call_and_return_conditional_losses_27346

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_27483

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
-__inference_ssquare_conv1_layer_call_fn_27235

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_ssquare_conv1_layer_call_and_return_conditional_losses_255552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_scol_conv2_layer_call_fn_27315

inputs!
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_scol_conv2_layer_call_and_return_conditional_losses_256912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_5_layer_call_fn_27510

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_258092
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
+__inference_concatenate_layer_call_fn_27584
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_258332
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????(:?????????:?????????:?????????<:?????????F:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????<
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????F
"
_user_specified_name
inputs/4
?W
?
__inference__traced_save_27845
file_prefix3
/savev2_ssquare_conv1_kernel_read_readvariableop1
-savev2_ssquare_conv1_bias_read_readvariableop3
/savev2_rsquare_conv1_kernel_read_readvariableop1
-savev2_rsquare_conv1_bias_read_readvariableop3
/savev2_ssquare_conv2_kernel_read_readvariableop1
-savev2_ssquare_conv2_bias_read_readvariableop0
,savev2_srow_conv2_kernel_read_readvariableop.
*savev2_srow_conv2_bias_read_readvariableop0
,savev2_scol_conv2_kernel_read_readvariableop.
*savev2_scol_conv2_bias_read_readvariableop/
+savev2_srow_conv_kernel_read_readvariableop-
)savev2_srow_conv_bias_read_readvariableop/
+savev2_scol_conv_kernel_read_readvariableop-
)savev2_scol_conv_bias_read_readvariableop3
/savev2_rsquare_conv2_kernel_read_readvariableop1
-savev2_rsquare_conv2_bias_read_readvariableop0
,savev2_rrow_conv2_kernel_read_readvariableop.
*savev2_rrow_conv2_bias_read_readvariableop0
,savev2_rcol_conv2_kernel_read_readvariableop.
*savev2_rcol_conv2_bias_read_readvariableop/
+savev2_rrow_conv_kernel_read_readvariableop-
)savev2_rrow_conv_bias_read_readvariableop/
+savev2_rcol_conv_kernel_read_readvariableop-
)savev2_rcol_conv_bias_read_readvariableop2
.savev2_sdense_layer_kernel_read_readvariableop0
,savev2_sdense_layer_bias_read_readvariableop2
.savev2_rdense_layer_kernel_read_readvariableop0
,savev2_rdense_layer_bias_read_readvariableop2
.savev2_score_output_kernel_read_readvariableop0
,savev2_score_output_bias_read_readvariableop3
/savev2_result_output_kernel_read_readvariableop1
-savev2_result_output_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_5_read_readvariableop&
"savev2_count_5_read_readvariableop&
"savev2_total_6_read_readvariableop&
"savev2_count_6_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*?
value?B?/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_ssquare_conv1_kernel_read_readvariableop-savev2_ssquare_conv1_bias_read_readvariableop/savev2_rsquare_conv1_kernel_read_readvariableop-savev2_rsquare_conv1_bias_read_readvariableop/savev2_ssquare_conv2_kernel_read_readvariableop-savev2_ssquare_conv2_bias_read_readvariableop,savev2_srow_conv2_kernel_read_readvariableop*savev2_srow_conv2_bias_read_readvariableop,savev2_scol_conv2_kernel_read_readvariableop*savev2_scol_conv2_bias_read_readvariableop+savev2_srow_conv_kernel_read_readvariableop)savev2_srow_conv_bias_read_readvariableop+savev2_scol_conv_kernel_read_readvariableop)savev2_scol_conv_bias_read_readvariableop/savev2_rsquare_conv2_kernel_read_readvariableop-savev2_rsquare_conv2_bias_read_readvariableop,savev2_rrow_conv2_kernel_read_readvariableop*savev2_rrow_conv2_bias_read_readvariableop,savev2_rcol_conv2_kernel_read_readvariableop*savev2_rcol_conv2_bias_read_readvariableop+savev2_rrow_conv_kernel_read_readvariableop)savev2_rrow_conv_bias_read_readvariableop+savev2_rcol_conv_kernel_read_readvariableop)savev2_rcol_conv_bias_read_readvariableop.savev2_sdense_layer_kernel_read_readvariableop,savev2_sdense_layer_bias_read_readvariableop.savev2_rdense_layer_kernel_read_readvariableop,savev2_rdense_layer_bias_read_readvariableop.savev2_score_output_kernel_read_readvariableop,savev2_score_output_bias_read_readvariableop/savev2_result_output_kernel_read_readvariableop-savev2_result_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_5_read_readvariableop"savev2_count_5_read_readvariableop"savev2_total_6_read_readvariableop"savev2_count_6_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *=
dtypes3
12/2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::::
:
:
:
:
:
:
:
:::
:
:
:
:
:
:
:
:	?d:d:	?d:d:d::d:: : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:,	(
&
_output_shapes
:
: 


_output_shapes
:
:,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:
: 

_output_shapes
:
:%!

_output_shapes
:	?d: 

_output_shapes
:d:%!

_output_shapes
:	?d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:d:  

_output_shapes
::!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: 
?
F
*__inference_flatten_10_layer_call_fn_27554

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_10_layer_call_and_return_conditional_losses_257612
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
`
D__inference_flatten_5_layer_call_and_return_conditional_losses_27505

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????F   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????F2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
E__inference_scol_conv2_layer_call_and_return_conditional_losses_25691

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_rdense_layer_layer_call_fn_27643

inputs
unknown:	?d
	unknown_0:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_rdense_layer_layer_call_and_return_conditional_losses_258462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
-__inference_concatenate_1_layer_call_fn_27603
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_258212
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????(:?????????:?????????:?????????<:?????????F:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????<
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????F
"
_user_specified_name
inputs/4
?
?
H__inference_rsquare_conv2_layer_call_and_return_conditional_losses_25640

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input6
serving_default_input:0?????????A
result_output0
StatefulPartitionedCall:0?????????@
score_output0
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer_with_weights-12
layer-25
layer_with_weights-13
layer-26
layer_with_weights-14
layer-27
layer_with_weights-15
layer-28
	optimizer
loss
 
signatures
#!_self_saveable_object_factories
"trainable_variables
#regularization_losses
$	variables
%	keras_api
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"??
_tf_keras_network??{"name": "c4net", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "c4net", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 6, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "ssquare_conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ssquare_conv1", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "rsquare_conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rsquare_conv1", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "ssquare_conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ssquare_conv2", "inbound_nodes": [[["ssquare_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "srow_conv2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "srow_conv2", "inbound_nodes": [[["ssquare_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "scol_conv2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "scol_conv2", "inbound_nodes": [[["ssquare_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "srow_conv", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "srow_conv", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "scol_conv", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "scol_conv", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "rsquare_conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rsquare_conv2", "inbound_nodes": [[["rsquare_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "rrow_conv2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rrow_conv2", "inbound_nodes": [[["rsquare_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "rcol_conv2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rcol_conv2", "inbound_nodes": [[["rsquare_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "rrow_conv", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rrow_conv", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "rcol_conv", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rcol_conv", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["ssquare_conv2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["srow_conv2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["scol_conv2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["srow_conv", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["scol_conv", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["rsquare_conv2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_8", "inbound_nodes": [[["rrow_conv2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_9", "inbound_nodes": [[["rcol_conv2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_10", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_10", "inbound_nodes": [[["rrow_conv", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_11", "inbound_nodes": [[["rcol_conv", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten_2", 0, 0, {}], ["flatten_3", 0, 0, {}], ["flatten_4", 0, 0, {}], ["flatten_5", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["flatten_7", 0, 0, {}], ["flatten_8", 0, 0, {}], ["flatten_9", 0, 0, {}], ["flatten_10", 0, 0, {}], ["flatten_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "sdense_layer", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "sdense_layer", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "rdense_layer", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rdense_layer", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "score_output", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "score_output", "inbound_nodes": [[["sdense_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "result_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "result_output", "inbound_nodes": [[["rdense_layer", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["score_output", 0, 0], ["result_output", 0, 0]]}, "shared_object_id": 61, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 7, 6, 2]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 6, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 7, 6, 2]}, "float32", "input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "c4net", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 6, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "ssquare_conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ssquare_conv1", "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "rsquare_conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rsquare_conv1", "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv2D", "config": {"name": "ssquare_conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ssquare_conv2", "inbound_nodes": [[["ssquare_conv1", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Conv2D", "config": {"name": "srow_conv2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "srow_conv2", "inbound_nodes": [[["ssquare_conv1", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Conv2D", "config": {"name": "scol_conv2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "scol_conv2", "inbound_nodes": [[["ssquare_conv1", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Conv2D", "config": {"name": "srow_conv", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "srow_conv", "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Conv2D", "config": {"name": "scol_conv", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "scol_conv", "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Conv2D", "config": {"name": "rsquare_conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rsquare_conv2", "inbound_nodes": [[["rsquare_conv1", 0, 0, {}]]], "shared_object_id": 24}, {"class_name": "Conv2D", "config": {"name": "rrow_conv2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rrow_conv2", "inbound_nodes": [[["rsquare_conv1", 0, 0, {}]]], "shared_object_id": 27}, {"class_name": "Conv2D", "config": {"name": "rcol_conv2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rcol_conv2", "inbound_nodes": [[["rsquare_conv1", 0, 0, {}]]], "shared_object_id": 30}, {"class_name": "Conv2D", "config": {"name": "rrow_conv", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rrow_conv", "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 33}, {"class_name": "Conv2D", "config": {"name": "rcol_conv", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rcol_conv", "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 36}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["ssquare_conv2", 0, 0, {}]]], "shared_object_id": 37}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["srow_conv2", 0, 0, {}]]], "shared_object_id": 38}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["scol_conv2", 0, 0, {}]]], "shared_object_id": 39}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["srow_conv", 0, 0, {}]]], "shared_object_id": 40}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["scol_conv", 0, 0, {}]]], "shared_object_id": 41}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["rsquare_conv2", 0, 0, {}]]], "shared_object_id": 42}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_8", "inbound_nodes": [[["rrow_conv2", 0, 0, {}]]], "shared_object_id": 43}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_9", "inbound_nodes": [[["rcol_conv2", 0, 0, {}]]], "shared_object_id": 44}, {"class_name": "Flatten", "config": {"name": "flatten_10", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_10", "inbound_nodes": [[["rrow_conv", 0, 0, {}]]], "shared_object_id": 45}, {"class_name": "Flatten", "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_11", "inbound_nodes": [[["rcol_conv", 0, 0, {}]]], "shared_object_id": 46}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten_2", 0, 0, {}], ["flatten_3", 0, 0, {}], ["flatten_4", 0, 0, {}], ["flatten_5", 0, 0, {}]]], "shared_object_id": 47}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["flatten_7", 0, 0, {}], ["flatten_8", 0, 0, {}], ["flatten_9", 0, 0, {}], ["flatten_10", 0, 0, {}], ["flatten_11", 0, 0, {}]]], "shared_object_id": 48}, {"class_name": "Dense", "config": {"name": "sdense_layer", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "sdense_layer", "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 51}, {"class_name": "Dense", "config": {"name": "rdense_layer", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rdense_layer", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]], "shared_object_id": 54}, {"class_name": "Dense", "config": {"name": "score_output", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 56}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "score_output", "inbound_nodes": [[["sdense_layer", 0, 0, {}]]], "shared_object_id": 57}, {"class_name": "Dense", "config": {"name": "result_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 58}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 59}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "result_output", "inbound_nodes": [[["rdense_layer", 0, 0, {}]]], "shared_object_id": 60}], "input_layers": [["input", 0, 0]], "output_layers": [["score_output", 0, 0], ["result_output", 0, 0]]}}, "training_config": {"loss": {"score_output": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 64}, "result_output": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 63}}, "metrics": [[{"class_name": "CategoricalCrossentropy", "config": {"name": "score_output_categorical_crossentropy", "dtype": "float32", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 65}, {"class_name": "CategoricalAccuracy", "config": {"name": "score_output_categorical_accuracy", "dtype": "float32"}, "shared_object_id": 66}], [{"class_name": "BinaryCrossentropy", "config": {"name": "result_output_binary_crossentropy", "dtype": "float32", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 67}, {"class_name": "BinaryAccuracy", "config": {"name": "result_output_binary_accuracy", "dtype": "float32", "threshold": 0.5}, "shared_object_id": 68}]], "weighted_metrics": null, "loss_weights": {"score_output": 1, "result_output": 1}, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
#&_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 6, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 6, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
?

'kernel
(bias
#)_self_saveable_object_factories
*trainable_variables
+regularization_losses
,	variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "ssquare_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "ssquare_conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 69}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 6, 2]}}
?

.kernel
/bias
#0_self_saveable_object_factories
1trainable_variables
2regularization_losses
3	variables
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "rsquare_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "rsquare_conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 70}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 6, 2]}}
?

5kernel
6bias
#7_self_saveable_object_factories
8trainable_variables
9regularization_losses
:	variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "ssquare_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "ssquare_conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["ssquare_conv1", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 20}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 2, 20]}}
?

<kernel
=bias
#>_self_saveable_object_factories
?trainable_variables
@regularization_losses
A	variables
B	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "srow_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "srow_conv2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["ssquare_conv1", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 20}}, "shared_object_id": 72}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 2, 20]}}
?

Ckernel
Dbias
#E_self_saveable_object_factories
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "scol_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "scol_conv2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["ssquare_conv1", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 20}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 2, 20]}}
?

Jkernel
Kbias
#L_self_saveable_object_factories
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "srow_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "srow_conv", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 74}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 6, 2]}}
?

Qkernel
Rbias
#S_self_saveable_object_factories
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "scol_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "scol_conv", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 75}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 6, 2]}}
?

Xkernel
Ybias
#Z_self_saveable_object_factories
[trainable_variables
\regularization_losses
]	variables
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "rsquare_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "rsquare_conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["rsquare_conv1", 0, 0, {}]]], "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 20}}, "shared_object_id": 76}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 2, 20]}}
?

_kernel
`bias
#a_self_saveable_object_factories
btrainable_variables
cregularization_losses
d	variables
e	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "rrow_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "rrow_conv2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["rsquare_conv1", 0, 0, {}]]], "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 20}}, "shared_object_id": 77}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 2, 20]}}
?

fkernel
gbias
#h_self_saveable_object_factories
itrainable_variables
jregularization_losses
k	variables
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "rcol_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "rcol_conv2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["rsquare_conv1", 0, 0, {}]]], "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 20}}, "shared_object_id": 78}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 2, 20]}}
?

mkernel
nbias
#o_self_saveable_object_factories
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "rrow_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "rrow_conv", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 79}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 6, 2]}}
?

tkernel
ubias
#v_self_saveable_object_factories
wtrainable_variables
xregularization_losses
y	variables
z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "rcol_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "rcol_conv", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 80}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 6, 2]}}
?
#{_self_saveable_object_factories
|trainable_variables
}regularization_losses
~	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["ssquare_conv2", 0, 0, {}]]], "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 81}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["srow_conv2", 0, 0, {}]]], "shared_object_id": 38, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 82}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["scol_conv2", 0, 0, {}]]], "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 83}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["srow_conv", 0, 0, {}]]], "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 84}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["scol_conv", 0, 0, {}]]], "shared_object_id": 41, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 85}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["rsquare_conv2", 0, 0, {}]]], "shared_object_id": 42, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 86}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["rrow_conv2", 0, 0, {}]]], "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 87}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["rcol_conv2", 0, 0, {}]]], "shared_object_id": 44, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 88}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_10", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["rrow_conv", 0, 0, {}]]], "shared_object_id": 45, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 89}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["rcol_conv", 0, 0, {}]]], "shared_object_id": 46, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 90}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten_2", 0, 0, {}], ["flatten_3", 0, 0, {}], ["flatten_4", 0, 0, {}], ["flatten_5", 0, 0, {}]]], "shared_object_id": 47, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 40]}, {"class_name": "TensorShape", "items": [null, 20]}, {"class_name": "TensorShape", "items": [null, 30]}, {"class_name": "TensorShape", "items": [null, 60]}, {"class_name": "TensorShape", "items": [null, 70]}]}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["flatten_7", 0, 0, {}], ["flatten_8", 0, 0, {}], ["flatten_9", 0, 0, {}], ["flatten_10", 0, 0, {}], ["flatten_11", 0, 0, {}]]], "shared_object_id": 48, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 40]}, {"class_name": "TensorShape", "items": [null, 20]}, {"class_name": "TensorShape", "items": [null, 30]}, {"class_name": "TensorShape", "items": [null, 60]}, {"class_name": "TensorShape", "items": [null, 70]}]}
?	
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "sdense_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "sdense_layer", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 51, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 220}}, "shared_object_id": 91}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 220]}}
?	
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "rdense_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "rdense_layer", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate_1", 0, 0, {}]]], "shared_object_id": 54, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 220}}, "shared_object_id": 92}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 220]}}
?	
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "score_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "score_output", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 56}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["sdense_layer", 0, 0, {}]]], "shared_object_id": 57, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 93}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?	
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "result_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "result_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 58}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 59}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["rdense_layer", 0, 0, {}]]], "shared_object_id": 60, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 94}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
"
	optimizer
 "
trackable_dict_wrapper
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
?
'0
(1
.2
/3
54
65
<6
=7
C8
D9
J10
K11
Q12
R13
X14
Y15
_16
`17
f18
g19
m20
n21
t22
u23
?24
?25
?26
?27
?28
?29
?30
?31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
'0
(1
.2
/3
54
65
<6
=7
C8
D9
J10
K11
Q12
R13
X14
Y15
_16
`17
f18
g19
m20
n21
t22
u23
?24
?25
?26
?27
?28
?29
?30
?31"
trackable_list_wrapper
?
 ?layer_regularization_losses
"trainable_variables
?layers
?non_trainable_variables
#regularization_losses
?layer_metrics
?metrics
$	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.:,2ssquare_conv1/kernel
 :2ssquare_conv1/bias
 "
trackable_dict_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
 ?layer_regularization_losses
*trainable_variables
?non_trainable_variables
+regularization_losses
?layer_metrics
?metrics
,	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,2rsquare_conv1/kernel
 :2rsquare_conv1/bias
 "
trackable_dict_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
 ?layer_regularization_losses
1trainable_variables
?non_trainable_variables
2regularization_losses
?layer_metrics
?metrics
3	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,2ssquare_conv2/kernel
 :2ssquare_conv2/bias
 "
trackable_dict_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
 ?layer_regularization_losses
8trainable_variables
?non_trainable_variables
9regularization_losses
?layer_metrics
?metrics
:	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)
2srow_conv2/kernel
:
2srow_conv2/bias
 "
trackable_dict_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
@regularization_losses
?layer_metrics
?metrics
A	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)
2scol_conv2/kernel
:
2scol_conv2/bias
 "
trackable_dict_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Ftrainable_variables
?non_trainable_variables
Gregularization_losses
?layer_metrics
?metrics
H	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
2srow_conv/kernel
:
2srow_conv/bias
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Mtrainable_variables
?non_trainable_variables
Nregularization_losses
?layer_metrics
?metrics
O	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
2scol_conv/kernel
:
2scol_conv/bias
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Ttrainable_variables
?non_trainable_variables
Uregularization_losses
?layer_metrics
?metrics
V	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,2rsquare_conv2/kernel
 :2rsquare_conv2/bias
 "
trackable_dict_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
?
 ?layer_regularization_losses
[trainable_variables
?non_trainable_variables
\regularization_losses
?layer_metrics
?metrics
]	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)
2rrow_conv2/kernel
:
2rrow_conv2/bias
 "
trackable_dict_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
?
 ?layer_regularization_losses
btrainable_variables
?non_trainable_variables
cregularization_losses
?layer_metrics
?metrics
d	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)
2rcol_conv2/kernel
:
2rcol_conv2/bias
 "
trackable_dict_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
?
 ?layer_regularization_losses
itrainable_variables
?non_trainable_variables
jregularization_losses
?layer_metrics
?metrics
k	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
2rrow_conv/kernel
:
2rrow_conv/bias
 "
trackable_dict_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
?
 ?layer_regularization_losses
ptrainable_variables
?non_trainable_variables
qregularization_losses
?layer_metrics
?metrics
r	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
2rcol_conv/kernel
:
2rcol_conv/bias
 "
trackable_dict_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
?
 ?layer_regularization_losses
wtrainable_variables
?non_trainable_variables
xregularization_losses
?layer_metrics
?metrics
y	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
|trainable_variables
?non_trainable_variables
}regularization_losses
?layer_metrics
?metrics
~	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$	?d2sdense_layer/kernel
:d2sdense_layer/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$	?d2rdense_layer/kernel
:d2rdense_layer/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#d2score_output/kernel
:2score_output/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$d2result_output/kernel
 :2result_output/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?metrics
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
X
?0
?1
?2
?3
?4
?5
?6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 95}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "score_output_loss", "dtype": "float32", "config": {"name": "score_output_loss", "dtype": "float32"}, "shared_object_id": 96}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "result_output_loss", "dtype": "float32", "config": {"name": "result_output_loss", "dtype": "float32"}, "shared_object_id": 97}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "CategoricalCrossentropy", "name": "score_output_categorical_crossentropy", "dtype": "float32", "config": {"name": "score_output_categorical_crossentropy", "dtype": "float32", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 65}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "CategoricalAccuracy", "name": "score_output_categorical_accuracy", "dtype": "float32", "config": {"name": "score_output_categorical_accuracy", "dtype": "float32"}, "shared_object_id": 66}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "BinaryCrossentropy", "name": "result_output_binary_crossentropy", "dtype": "float32", "config": {"name": "result_output_binary_crossentropy", "dtype": "float32", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 67}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "BinaryAccuracy", "name": "result_output_binary_accuracy", "dtype": "float32", "config": {"name": "result_output_binary_accuracy", "dtype": "float32", "threshold": 0.5}, "shared_object_id": 68}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
?2?
@__inference_c4net_layer_call_and_return_conditional_losses_26932
@__inference_c4net_layer_call_and_return_conditional_losses_27073
@__inference_c4net_layer_call_and_return_conditional_losses_26621
@__inference_c4net_layer_call_and_return_conditional_losses_26718?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_25520?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *,?)
'?$
input?????????
?2?
%__inference_c4net_layer_call_fn_25974
%__inference_c4net_layer_call_fn_27144
%__inference_c4net_layer_call_fn_27215
%__inference_c4net_layer_call_fn_26524?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_ssquare_conv1_layer_call_and_return_conditional_losses_27226?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_ssquare_conv1_layer_call_fn_27235?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_rsquare_conv1_layer_call_and_return_conditional_losses_27246?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_rsquare_conv1_layer_call_fn_27255?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_ssquare_conv2_layer_call_and_return_conditional_losses_27266?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_ssquare_conv2_layer_call_fn_27275?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_srow_conv2_layer_call_and_return_conditional_losses_27286?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_srow_conv2_layer_call_fn_27295?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_scol_conv2_layer_call_and_return_conditional_losses_27306?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_scol_conv2_layer_call_fn_27315?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_srow_conv_layer_call_and_return_conditional_losses_27326?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_srow_conv_layer_call_fn_27335?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_scol_conv_layer_call_and_return_conditional_losses_27346?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_scol_conv_layer_call_fn_27355?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_rsquare_conv2_layer_call_and_return_conditional_losses_27366?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_rsquare_conv2_layer_call_fn_27375?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_rrow_conv2_layer_call_and_return_conditional_losses_27386?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_rrow_conv2_layer_call_fn_27395?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_rcol_conv2_layer_call_and_return_conditional_losses_27406?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_rcol_conv2_layer_call_fn_27415?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_rrow_conv_layer_call_and_return_conditional_losses_27426?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_rrow_conv_layer_call_fn_27435?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_rcol_conv_layer_call_and_return_conditional_losses_27446?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_rcol_conv_layer_call_fn_27455?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_1_layer_call_and_return_conditional_losses_27461?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_1_layer_call_fn_27466?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_2_layer_call_and_return_conditional_losses_27472?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_2_layer_call_fn_27477?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_3_layer_call_and_return_conditional_losses_27483?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_3_layer_call_fn_27488?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_4_layer_call_and_return_conditional_losses_27494?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_4_layer_call_fn_27499?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_5_layer_call_and_return_conditional_losses_27505?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_5_layer_call_fn_27510?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_7_layer_call_and_return_conditional_losses_27516?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_7_layer_call_fn_27521?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_8_layer_call_and_return_conditional_losses_27527?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_8_layer_call_fn_27532?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_9_layer_call_and_return_conditional_losses_27538?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_9_layer_call_fn_27543?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_10_layer_call_and_return_conditional_losses_27549?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_10_layer_call_fn_27554?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_11_layer_call_and_return_conditional_losses_27560?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_11_layer_call_fn_27565?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_concatenate_layer_call_and_return_conditional_losses_27575?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_concatenate_layer_call_fn_27584?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_27594?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_1_layer_call_fn_27603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_sdense_layer_layer_call_and_return_conditional_losses_27614?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_sdense_layer_layer_call_fn_27623?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_rdense_layer_layer_call_and_return_conditional_losses_27634?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_rdense_layer_layer_call_fn_27643?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_score_output_layer_call_and_return_conditional_losses_27654?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_score_output_layer_call_fn_27663?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_result_output_layer_call_and_return_conditional_losses_27674?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_result_output_layer_call_fn_27683?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_26791input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_25520?(./'(tumnfg_`XYQRJKCD<=56????????6?3
,?)
'?$
input?????????
? "u?r
8
result_output'?$
result_output?????????
6
score_output&?#
score_output??????????
@__inference_c4net_layer_call_and_return_conditional_losses_26621?(./'(tumnfg_`XYQRJKCD<=56????????>?;
4?1
'?$
input?????????
p 

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
@__inference_c4net_layer_call_and_return_conditional_losses_26718?(./'(tumnfg_`XYQRJKCD<=56????????>?;
4?1
'?$
input?????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
@__inference_c4net_layer_call_and_return_conditional_losses_26932?(./'(tumnfg_`XYQRJKCD<=56??????????<
5?2
(?%
inputs?????????
p 

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
@__inference_c4net_layer_call_and_return_conditional_losses_27073?(./'(tumnfg_`XYQRJKCD<=56??????????<
5?2
(?%
inputs?????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
%__inference_c4net_layer_call_fn_25974?(./'(tumnfg_`XYQRJKCD<=56????????>?;
4?1
'?$
input?????????
p 

 
? "=?:
?
0?????????
?
1??????????
%__inference_c4net_layer_call_fn_26524?(./'(tumnfg_`XYQRJKCD<=56????????>?;
4?1
'?$
input?????????
p

 
? "=?:
?
0?????????
?
1??????????
%__inference_c4net_layer_call_fn_27144?(./'(tumnfg_`XYQRJKCD<=56??????????<
5?2
(?%
inputs?????????
p 

 
? "=?:
?
0?????????
?
1??????????
%__inference_c4net_layer_call_fn_27215?(./'(tumnfg_`XYQRJKCD<=56??????????<
5?2
(?%
inputs?????????
p

 
? "=?:
?
0?????????
?
1??????????
H__inference_concatenate_1_layer_call_and_return_conditional_losses_27594????
???
???
"?
inputs/0?????????(
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????<
"?
inputs/4?????????F
? "&?#
?
0??????????
? ?
-__inference_concatenate_1_layer_call_fn_27603????
???
???
"?
inputs/0?????????(
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????<
"?
inputs/4?????????F
? "????????????
F__inference_concatenate_layer_call_and_return_conditional_losses_27575????
???
???
"?
inputs/0?????????(
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????<
"?
inputs/4?????????F
? "&?#
?
0??????????
? ?
+__inference_concatenate_layer_call_fn_27584????
???
???
"?
inputs/0?????????(
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????<
"?
inputs/4?????????F
? "????????????
E__inference_flatten_10_layer_call_and_return_conditional_losses_27549`7?4
-?*
(?%
inputs?????????

? "%?"
?
0?????????<
? ?
*__inference_flatten_10_layer_call_fn_27554S7?4
-?*
(?%
inputs?????????

? "??????????<?
E__inference_flatten_11_layer_call_and_return_conditional_losses_27560`7?4
-?*
(?%
inputs?????????

? "%?"
?
0?????????F
? ?
*__inference_flatten_11_layer_call_fn_27565S7?4
-?*
(?%
inputs?????????

? "??????????F?
D__inference_flatten_1_layer_call_and_return_conditional_losses_27461`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????(
? ?
)__inference_flatten_1_layer_call_fn_27466S7?4
-?*
(?%
inputs?????????
? "??????????(?
D__inference_flatten_2_layer_call_and_return_conditional_losses_27472`7?4
-?*
(?%
inputs?????????

? "%?"
?
0?????????
? ?
)__inference_flatten_2_layer_call_fn_27477S7?4
-?*
(?%
inputs?????????

? "???????????
D__inference_flatten_3_layer_call_and_return_conditional_losses_27483`7?4
-?*
(?%
inputs?????????

? "%?"
?
0?????????
? ?
)__inference_flatten_3_layer_call_fn_27488S7?4
-?*
(?%
inputs?????????

? "???????????
D__inference_flatten_4_layer_call_and_return_conditional_losses_27494`7?4
-?*
(?%
inputs?????????

? "%?"
?
0?????????<
? ?
)__inference_flatten_4_layer_call_fn_27499S7?4
-?*
(?%
inputs?????????

? "??????????<?
D__inference_flatten_5_layer_call_and_return_conditional_losses_27505`7?4
-?*
(?%
inputs?????????

? "%?"
?
0?????????F
? ?
)__inference_flatten_5_layer_call_fn_27510S7?4
-?*
(?%
inputs?????????

? "??????????F?
D__inference_flatten_7_layer_call_and_return_conditional_losses_27516`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????(
? ?
)__inference_flatten_7_layer_call_fn_27521S7?4
-?*
(?%
inputs?????????
? "??????????(?
D__inference_flatten_8_layer_call_and_return_conditional_losses_27527`7?4
-?*
(?%
inputs?????????

? "%?"
?
0?????????
? ?
)__inference_flatten_8_layer_call_fn_27532S7?4
-?*
(?%
inputs?????????

? "???????????
D__inference_flatten_9_layer_call_and_return_conditional_losses_27538`7?4
-?*
(?%
inputs?????????

? "%?"
?
0?????????
? ?
)__inference_flatten_9_layer_call_fn_27543S7?4
-?*
(?%
inputs?????????

? "???????????
E__inference_rcol_conv2_layer_call_and_return_conditional_losses_27406lfg7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????

? ?
*__inference_rcol_conv2_layer_call_fn_27415_fg7?4
-?*
(?%
inputs?????????
? " ??????????
?
D__inference_rcol_conv_layer_call_and_return_conditional_losses_27446ltu7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????

? ?
)__inference_rcol_conv_layer_call_fn_27455_tu7?4
-?*
(?%
inputs?????????
? " ??????????
?
G__inference_rdense_layer_layer_call_and_return_conditional_losses_27634_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? ?
,__inference_rdense_layer_layer_call_fn_27643R??0?-
&?#
!?
inputs??????????
? "??????????d?
H__inference_result_output_layer_call_and_return_conditional_losses_27674^??/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? ?
-__inference_result_output_layer_call_fn_27683Q??/?,
%?"
 ?
inputs?????????d
? "???????????
E__inference_rrow_conv2_layer_call_and_return_conditional_losses_27386l_`7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????

? ?
*__inference_rrow_conv2_layer_call_fn_27395__`7?4
-?*
(?%
inputs?????????
? " ??????????
?
D__inference_rrow_conv_layer_call_and_return_conditional_losses_27426lmn7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????

? ?
)__inference_rrow_conv_layer_call_fn_27435_mn7?4
-?*
(?%
inputs?????????
? " ??????????
?
H__inference_rsquare_conv1_layer_call_and_return_conditional_losses_27246l./7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
-__inference_rsquare_conv1_layer_call_fn_27255_./7?4
-?*
(?%
inputs?????????
? " ???????????
H__inference_rsquare_conv2_layer_call_and_return_conditional_losses_27366lXY7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
-__inference_rsquare_conv2_layer_call_fn_27375_XY7?4
-?*
(?%
inputs?????????
? " ???????????
E__inference_scol_conv2_layer_call_and_return_conditional_losses_27306lCD7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????

? ?
*__inference_scol_conv2_layer_call_fn_27315_CD7?4
-?*
(?%
inputs?????????
? " ??????????
?
D__inference_scol_conv_layer_call_and_return_conditional_losses_27346lQR7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????

? ?
)__inference_scol_conv_layer_call_fn_27355_QR7?4
-?*
(?%
inputs?????????
? " ??????????
?
G__inference_score_output_layer_call_and_return_conditional_losses_27654^??/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? ?
,__inference_score_output_layer_call_fn_27663Q??/?,
%?"
 ?
inputs?????????d
? "???????????
G__inference_sdense_layer_layer_call_and_return_conditional_losses_27614_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? ?
,__inference_sdense_layer_layer_call_fn_27623R??0?-
&?#
!?
inputs??????????
? "??????????d?
#__inference_signature_wrapper_26791?(./'(tumnfg_`XYQRJKCD<=56??????????<
? 
5?2
0
input'?$
input?????????"u?r
8
result_output'?$
result_output?????????
6
score_output&?#
score_output??????????
E__inference_srow_conv2_layer_call_and_return_conditional_losses_27286l<=7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????

? ?
*__inference_srow_conv2_layer_call_fn_27295_<=7?4
-?*
(?%
inputs?????????
? " ??????????
?
D__inference_srow_conv_layer_call_and_return_conditional_losses_27326lJK7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????

? ?
)__inference_srow_conv_layer_call_fn_27335_JK7?4
-?*
(?%
inputs?????????
? " ??????????
?
H__inference_ssquare_conv1_layer_call_and_return_conditional_losses_27226l'(7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
-__inference_ssquare_conv1_layer_call_fn_27235_'(7?4
-?*
(?%
inputs?????????
? " ???????????
H__inference_ssquare_conv2_layer_call_and_return_conditional_losses_27266l567?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
-__inference_ssquare_conv2_layer_call_fn_27275_567?4
-?*
(?%
inputs?????????
? " ??????????