.class public Lcom/example/lab5_2/MainActivity;
.super Landroidx/appcompat/app/AppCompatActivity;
.source "MainActivity.java"


# static fields
.field private static final DUMMY_CREDENTIALS:[Ljava/lang/String;


# instance fields
.field private password:Landroid/widget/EditText;

.field private username:Landroid/widget/EditText;


# direct methods
.method static constructor <clinit>()V
    .registers 3

    .line 21
    const/4 v0, 0x2

    new-array v0, v0, [Ljava/lang/String;

    const/4 v1, 0x0

    const-string v2, "foo@example.com:111"

    aput-object v2, v0, v1

    const/4 v1, 0x1

    const-string v2, "bar@example.com:222"

    aput-object v2, v0, v1

    sput-object v0, Lcom/example/lab5_2/MainActivity;->DUMMY_CREDENTIALS:[Ljava/lang/String;

    return-void
.end method

.method public constructor <init>()V
    .registers 1

    .line 17
    invoke-direct {p0}, Landroidx/appcompat/app/AppCompatActivity;-><init>()V

    return-void
.end method


# virtual methods
.method public loginClick(Landroid/view/View;)V
    .registers 15
    .param p1, "view"    # Landroid/view/View;

    .line 56
    invoke-virtual {p0}, Lcom/example/lab5_2/MainActivity;->getLayoutInflater()Landroid/view/LayoutInflater;

    move-result-object v0

    sget v1, Lcom/example/lab5_2/R$layout;->popup_view:I

    const/4 v2, 0x0

    invoke-virtual {v0, v1, v2}, Landroid/view/LayoutInflater;->inflate(ILandroid/view/ViewGroup;)Landroid/view/View;

    move-result-object v0

    .line 57
    .local v0, "popupView":Landroid/view/View;
    new-instance v1, Landroid/widget/PopupWindow;

    const/4 v2, -0x2

    const/4 v3, 0x1

    invoke-direct {v1, v0, v2, v2, v3}, Landroid/widget/PopupWindow;-><init>(Landroid/view/View;IIZ)V

    .line 60
    .local v1, "popupWindow":Landroid/widget/PopupWindow;
    sget v2, Lcom/example/lab5_2/R$id;->username:I

    invoke-virtual {p0, v2}, Lcom/example/lab5_2/MainActivity;->findViewById(I)Landroid/view/View;

    move-result-object v2

    check-cast v2, Landroid/widget/EditText;

    iput-object v2, p0, Lcom/example/lab5_2/MainActivity;->username:Landroid/widget/EditText;

    .line 61
    sget v2, Lcom/example/lab5_2/R$id;->password:I

    invoke-virtual {p0, v2}, Lcom/example/lab5_2/MainActivity;->findViewById(I)Landroid/view/View;

    move-result-object v2

    check-cast v2, Landroid/widget/EditText;

    iput-object v2, p0, Lcom/example/lab5_2/MainActivity;->password:Landroid/widget/EditText;

    .line 63
    iget-object v2, p0, Lcom/example/lab5_2/MainActivity;->username:Landroid/widget/EditText;

    invoke-virtual {v2}, Landroid/widget/EditText;->getText()Landroid/text/Editable;

    move-result-object v2

    invoke-virtual {v2}, Ljava/lang/Object;->toString()Ljava/lang/String;

    move-result-object v2

    .line 64
    .local v2, "un":Ljava/lang/String;
    iget-object v4, p0, Lcom/example/lab5_2/MainActivity;->password:Landroid/widget/EditText;

    invoke-virtual {v4}, Landroid/widget/EditText;->getText()Landroid/text/Editable;

    move-result-object v4

    invoke-virtual {v4}, Ljava/lang/Object;->toString()Ljava/lang/String;

    move-result-object v4

    .line 65
    .local v4, "pw":Ljava/lang/String;
    new-instance v5, Ljava/lang/StringBuilder;

    invoke-direct {v5}, Ljava/lang/StringBuilder;-><init>()V

    const-string v6, "input"

    invoke-virtual {v5, v6}, Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;

    move-result-object v5

    invoke-virtual {v5, v2}, Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;

    move-result-object v5

    invoke-virtual {v5, v4}, Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;

    move-result-object v5

    invoke-virtual {v5}, Ljava/lang/StringBuilder;->toString()Ljava/lang/String;

    move-result-object v5

    const-string v6, "zsc"

    invoke-static {v6, v5}, Landroid/util/Log;->e(Ljava/lang/String;Ljava/lang/String;)I

    .line 66
    sget-object v5, Lcom/example/lab5_2/MainActivity;->DUMMY_CREDENTIALS:[Ljava/lang/String;

    array-length v7, v5

    const/4 v8, 0x0

    move v9, v8

    :goto_5b
    if-ge v9, v7, :cond_95

    aget-object v10, v5, v9

    .line 67
    .local v10, "s":Ljava/lang/String;
    const-string v11, ":"

    invoke-virtual {v10, v11}, Ljava/lang/String;->split(Ljava/lang/String;)[Ljava/lang/String;

    move-result-object v12

    aget-object v12, v12, v8

    invoke-virtual {v2, v12}, Ljava/lang/String;->equals(Ljava/lang/Object;)Z

    move-result v12

    if-eqz v12, :cond_92

    invoke-virtual {v10, v11}, Ljava/lang/String;->split(Ljava/lang/String;)[Ljava/lang/String;

    move-result-object v11

    aget-object v11, v11, v3

    invoke-virtual {v4, v11}, Ljava/lang/String;->equals(Ljava/lang/Object;)Z

    move-result v11

    if-eqz v11, :cond_92

    .line 68
    new-instance v11, Ljava/lang/StringBuilder;

    invoke-direct {v11}, Ljava/lang/StringBuilder;-><init>()V

    invoke-virtual {v11, v2}, Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;

    move-result-object v11

    const-string v12, " log in successfully"

    invoke-virtual {v11, v12}, Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;

    move-result-object v11

    invoke-virtual {v11}, Ljava/lang/StringBuilder;->toString()Ljava/lang/String;

    move-result-object v11

    invoke-static {v6, v11}, Landroid/util/Log;->e(Ljava/lang/String;Ljava/lang/String;)I

    .line 69
    invoke-virtual {v1, p1}, Landroid/widget/PopupWindow;->showAsDropDown(Landroid/view/View;)V

    .line 66
    .end local v10    # "s":Ljava/lang/String;
    :cond_92
    add-int/lit8 v9, v9, 0x1

    goto :goto_5b

    .line 74
    :cond_95
    return-void
.end method

.method protected onCreate(Landroid/os/Bundle;)V
    .registers 3
    .param p1, "savedInstanceState"    # Landroid/os/Bundle;

    .line 27
    invoke-super {p0, p1}, Landroidx/appcompat/app/AppCompatActivity;->onCreate(Landroid/os/Bundle;)V

    .line 28
    invoke-static {p0}, Landroidx/activity/EdgeToEdge;->enable(Landroidx/activity/ComponentActivity;)V

    .line 29
    sget v0, Lcom/example/lab5_2/R$layout;->activity_main:I

    invoke-virtual {p0, v0}, Lcom/example/lab5_2/MainActivity;->setContentView(I)V

    .line 53
    return-void
.end method
