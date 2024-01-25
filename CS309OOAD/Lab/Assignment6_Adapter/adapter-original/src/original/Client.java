package original;

import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;
import java.util.Scanner;

public class Client {

    public static void main(String[] args) {
        List<StaffModel> list = new ArrayList<>();
        FileOperateInterfaceV1 adaptee = new FileOperate();
        ManageStaffInterface adaptee2 = new ManageStaff();
        FileOperateInterfaceV2 fileOperator = new Adapter(adaptee, adaptee2);
        Scanner input = new Scanner(System.in);
        System.out.println("Please select operation: 1.readFile 2.listFile 3.writeByName 4.writeByRoom 5.Add Staff 6.Remove Staff:");
        int op = 0;
        do {
            try {
                op = input.nextInt();
                switch (op) {
                    case 1:
                        list = fileOperator.readAllStaff();
                        break;
                    case 2:
                        fileOperator.listAllStaff(list);
                        break;
                    case 3:
                        fileOperator.writeByName(list);
                        break;
                    case 4:
                        fileOperator.writeByRoom(list);
                        break;
                    case 5:
                        fileOperator.addNewStaff(list);
                        break;
                    case 6:
                        fileOperator.removeStaffByName(list);
                        break;
                }
            } catch (InputMismatchException e) {
                System.out.println("Exception:" + e);
                input.nextLine();
            }
        } while (op != 0);
        input.close();
    }
}
