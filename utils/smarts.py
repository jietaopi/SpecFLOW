FG_SMARTS = {
    "Alkane": "[CX4]",
    "Alkene": "[CX3]=[CX3]",
    "Alkyne": "[CX2]#C",
    "Arene": "[$([cX3](:*):*),$([cX2+](:*):*)]",
    "Alcohol": "[#6][OX2H]",
    "Ether": "[OD2]([#6])[#6]",
    "Aldehyde": "[CX3H1](=O)[#6]",
    "Ketone": "[#6][CX3](=O)[#6]",
    "Carboxylicacid": "[CX3](=O)[OX2H1]",
    "Ester": "[#6][CX3](=O)[OX2H0][#6]",
    "haloalkane": "[#6][F,Cl,Br,I]",
    "Alkylhalide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "Amine": "[NX3;$(NC=O)]",
    "Amide": "[NX3][CX3](=[OX1])[#6]",
    "Nitrile": "[NX1]#[CX2]",
    "Sulfide": "[#16X2H0]",
    "Thiol": "[#16X2H]"
}


FG_NAMES = list(FG_SMARTS.keys())