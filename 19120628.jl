#Hoàng Anh Quân
#19120628

###Nếu đã cài đặt 2 gói này thì có thể commet lại rồi chạy chương trình####
import Pkg
Pkg.add("DataFrames")
Pkg.add("CSV")


using CSV, DataFrames, Random

#ĐỊNH NGHĨA HẰNG 
const MINIMUM_SIZE = 5::Int64

#Định nghĩa các node của một cây quyết định có 2 trường giá trị >cutoff và <= cutoff 
mutable struct TreeNode
    #ATTRUBUTE
    Name::String
    cutoff::Union{Float64,Int64}
    leftnode::Union{TreeNode,Nothing}
    rightnode::Union{TreeNode,Nothing}

end


#In ra cây quyết định ra màn hình
function printTree(Root::Union{TreeNode,Nothing}, x = 2, eq = "")
    if Root === nothing
        return
    end
    len = sizeof(Root.Name) + 5
    printTree(Root.leftnode, x + len, "(<=)")

    if (Root.cutoff == 0)
        println(" "^x, eq, "-", Root.Name)
    else
        println(" "^x, eq, "-", Root.Name, " (", Root.cutoff, ")")
    end
    printTree(Root.rightnode, x + len, "(>)")

end


#Đọc các dữ liệu tập Test, Training từ tập iris.csv, 
# với Test và Training được lấy ngẫu nhiên với tỷ là 1/3 và 2/3
function Read_IRIS_CSV()
    df_iris = CSV.read("IRIS.csv", DataFrame)
    nRow = nrow(df_iris)#lấy số dòng dữ liệu của tập iris
    nRowTest = floor(Int, (1 / 3) * nRow)
    Indexs = randperm(nRow)[1:nRowTest] #ramdom giá trị từ 1->nRow, và có nRowTest giá trị   index các dong của tập Test
    #Tập test: 

    dfTest = df_iris[Indexs, :]#1/3 dữ liệu tập iris là test set
    #Tập Training:
    dfTrain = df_iris[Not(Indexs), :]#2/3 dữ liệu tập iris là training set

    # val_species=unique(df_iris.species)
    return dfTrain, dfTest
end

#Tính giá trịnh entrypy
function Entropy(dataset::DataFrame)
    val_class = unique(dataset[!, end])
    nData = nrow(dataset)
    H = 0
    for val in val_class
        num = count(x -> (x == val), dataset[!, end])
        p = num / nData
        if (p == 0)
            continue
        end
        H -= p * log2(p)
    end
    return H
end

#Tìm các giá trị chỉ số dòng của dataset có attribute x <=cutoff
function getIndex(dataset::DataFrame, Atribute::String, cutoff::Float64)
    Indexs = []
    x = 1
    for i in dataset[!, Atribute]
        if (i <= cutoff)
            append!(Indexs, x)
        end
        x += 1
    end
    return Indexs
end


#Tính entropy trung bình của một thuộc tính
function EntropyAVG(dataset::DataFrame, Atribute_X::String, cutoff::Union{Float64,Int64})
    #Chia dữ liệu dataset thành 2 phần: data1:<=cutoff và data2: >cutoff
    Index = getIndex(dataset, Atribute_X, cutoff) #get index row atribute <=cutoff
    smallerData = dataset[Index, :]      #data[Atribute_X] <=cutoff
    greaterData = dataset[Not(Index), :] #data[Atribute_X] >cutoff
    ndata = nrow(dataset)
    ndata1 = nrow(smallerData)
    ndata2 = nrow(greaterData)
    H1 = (ndata1 / ndata) * Entropy(smallerData)
    H2 = (ndata2 / ndata) * Entropy(greaterData)
    return H1 + H2
end


#Tìm giá trị cutoff của một thuộc tính sao cho chỉ số entrypy trung bình của thuộc tính đó là nhỏ nhất
function FindCutOff(dataset::DataFrame, colName::String)
    listCutOff = sort(unique(dataset[!, colName]))
    minEntropy = typemax(Int)
    cutoff = 0
    for Cut in listCutOff
        H = EntropyAVG(dataset, colName, Cut)
        if (minEntropy > H)
            minEntropy = H
            cutoff = Cut
        end
    end
    return cutoff, minEntropy
end



#Tính hệ số Information Gian của một thuộc tính
function InfoGain(dataset::DataFrame, Atribute_X::String, cutoff)
    return Entropy(dataset) - EntropyAVG(dataset, Atribute_X, cutoff)
end


function BestInfoGain(dataset::DataFrame)
    #INFO_GAIN= E(data)-E(col,data) =>INFO_GAIN(max) -> E(col,data)(min)

    AttrList = names(dataset)[1:4]
    Min = typemax(Int)
    BestAtribute = "Empty"
    cutoff = 0
    for atribute in AttrList
        cut, minEntropy = FindCutOff(dataset, atribute)
        if (minEntropy < Min)
            Min = minEntropy
            cutoff = cut
            BestAtribute = atribute
        end
    end
    return BestAtribute, cutoff
end



#Thuật toán tìm cây quyết định:
#Điều kiện dừng: +Nếu node đó có entropy bằng 0, tức mọi điểm trong node đều thuộc một class.
function DecisionTree(dataset::DataFrame, Root::TreeNode)
    #Nếu dataset tinh khiết thì return
    if (Entropy(dataset) == 0)
        Root.Name = dataset[1, end]
        return
    end

    #Tìm tên thuộc tính và cufoff sao cho giá trị Info Gain đạt lớn nhât
    choiceAttr, cutoff = BestInfoGain(dataset)
    Root.Name = choiceAttr
    Root.cutoff = cutoff

    #Lọc data chia thành 2 phần có giá trị <= cutoff và > cutoff
    Index = getIndex(dataset, choiceAttr, cutoff) #get index row atribute <=cutoff
    smallerData = dataset[Index, :]      #data[Atribute_X] <=cutoff
    greaterData = dataset[Not(Index), :] #data[Atribute_X] >cutoff

    #Gọi đệ qui để tiếp tục tục tìm node tiếp theo
    Root.leftnode = TreeNode("", 0, nothing, nothing)
    DecisionTree(smallerData, Root.leftnode)
    Root.rightnode = TreeNode("", 0, nothing, nothing)
    DecisionTree(greaterData, Root.rightnode)

end


#Tìm giá trị ước tính dựa vào cây quyết định
function getClassValBaseOnDT3(Root::Union{TreeNode,Nothing}, RowData::DataFrameRow)
    if Root.cutoff == 0
        return Root.Name
    end
    value = RowData[Root.Name]
    if value <= Root.cutoff
        return getClassValBaseOnDT3(Root.leftnode, RowData)
    else
        return getClassValBaseOnDT3(Root.rightnode, RowData)
    end
end

#Đánh giá độ chính xác của cây quyết định dựa vào tập test
function getAccuracy(DataTest::DataFrame, Roots::TreeNode)
    nData = nrow(DataTest)
    count = 0

    for i in 1:nData
        Rowdf = DataTest[i, :]
        Species = getClassValBaseOnDT3(Roots, Rowdf)
        if (Species == Rowdf[end])
            count += 1
        end
    end
    return count / nData * 100
end




#///==========================MAIN============================\\\
#main funtion 
function main()

    dfTrain, dfTest = Read_IRIS_CSV()
    println("Data Training Set:", nrow(dfTrain))
    println("Data Test Set:", nrow(dfTest))
    #định nghĩa gốc của cây quết định, và chạy hàm DecisionTree để tìm các node của cây
    Roots = TreeNode("", 0, nothing, nothing)
    DecisionTree(dfTrain, Roots) #5:

    #in cây quết định ra màn hình
    println("\n\n||========IN CAY QUYET DINH (DESISION TREE)=======||\n")
    printTree(Roots)

    #in độ chính xác ra màn hình
    acc = getAccuracy(dfTest, Roots)

    println("\n\n||===========DO CHINH XAC: ", acc, "%")

end

#chạy hàm main 
#test 1: 
println("TEST 1:\n")
main()

#test 2: 
println("TEST 2:\n")
main()

#test 3: 
println("TEST 3:\n")
main()
