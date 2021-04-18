|名称|类型|描述|
|-|-|-|
|container|DomNode|heatmap载体，必须|
|backgroundColor|String|canvas背景色|
|gradient|Object|渐变色，用于制作调色板|
|radius|Number|点半径|
|opacity|Number[0,1]|热力图渲染透明度，设置该参数后，maxOpaticy、minOpaticy参数无效|
|maxOpacity|Number[0,1]|热力图渲染允许的透明度的最大值，默认为1|
|minOpacity|Number[0,1]|热力图渲染允许的透明度的最小值，默认为0|
|useGradientOpacity|Boolean|热力图渲染是否使用gradient渐变色的透明度，默认是false，设置为true的情况下opacity，maxOpacity，minOpacity无效。建议不设置该参数|
|onExtremaChange|function callback|渲染的数据极值发生变化的事件回调函数,由以下函数触发setData、setDataMax. setDataMin。可用于图例的展示，2.0版本中不再在库中内置图例展示。|
|blur|Number[0,1]|模糊因子，适用于所有的数据点，默认为0.35。模糊因子越高，梯度就越平滑。也就是做放射颜色渐变时的内圆半径越小|
|xField|String|x坐标属性名称,默认为'x'|
|yField|String|y坐标属性名称,默认为'y'|
|valueField|String|渲染属性名称,默认为'value' ,若渲染属性不存在,则一个点value为1|
|defaul tRadius|40|点半径|
|backgroundColor|canvas2d|绘图工具，canvas绘图对象，因canvas现在只支持2D绘图,这里应是作为预留拓展配置。|
|defaultGradient|{0.25:"rgb(0,0,255)",<br>0.55:"rgb(0,255,0)",<br>0.85:“yellow",<br>1.0:"rgb(255,0,0)"}|渐变色，用于制作调色板|
|defaultMaxOpacity|1|透明度最大值|
|defaultMinOpacity|0|透明度最小值|
|defaultBlur|0.85|颜色渐变因子值，值越大，内院越小，热力效果越小。<br>ps:1-defaultBlur的值作为CanvasRedneringContext2D.createLinearGradient()方法的第三个实参，即渐变开始圆的半径|
|defaultXField|x|x坐标属性名称|
|defaultYFiedl|y|y坐标属性名称|
|defaultValueField|value|渲染字段名称|
|setData|Object[] data|设置渲染的数据|
|addData|Object[] data|添加渲染的数据|
|removeData||溢出渲染的数据，预留的方法，尚未实现|
|setDataMax|Number|设置渲染的最大值，并重新绘制|
|setDataMin|Number|设置渲染的最小值，并重新绘制|
|configure|config|设置热力图配置信息|
|repaint||重绘|
|getData||返回store中的数据|
|getDataURL||返回热力图图片|
|getValueAt|point|返回一个像素点渲染数据|
|_colorize||对绘制完成的渲染数据进行着色|
|renderPartial|data|对传入的数据进行渲染|
|renderAll|data|将传入的数据作为全部数据进行渲染，即会先清除已有的渲染数据|
|_updateGradient|config|更新渲染的渐变色，即重新设置调色板|
|updateConfig|config|更新渲染配置信息，即更新渲染的渐变色和渲染的相关样式|
|setDimension|width,height|设置画布宽高|
|_clear||清除画布|
|getValueAt|point|获取一个像素点数据，包括颜色及透明度|
|Canvas2dRenderer|config|渲染器初始化方法|
|_getColorPalette|config|获取调色板|
|_setStyles|config|设置渲染样式，背景色，渐变因子，透明度|
|_prepareData|data(数据为store已处理完成的)|将store处理完成的数据转换为可渲染的数据|
|_getPointTemplate|radius，半径，blur，渐变因子|根据圆半径和渐变因子确定一个点渲染的模板|
|_drawAlpha|data|根据渲染的数据进行点绘制，未着色|
