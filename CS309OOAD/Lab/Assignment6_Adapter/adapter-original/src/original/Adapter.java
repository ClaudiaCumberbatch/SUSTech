package original;

import java.util.List;

public class Adapter implements FileOperateInterfaceV2 {
    FileOperateInterfaceV1 adaptee;
    ManageStaffInterface adaptee2;
    public Adapter(FileOperateInterfaceV1 adaptee, ManageStaffInterface adaptee2) {
        this.adaptee = adaptee;
        this.adaptee2 = adaptee2;
    }

    @Override
    public List<original.StaffModel> readAllStaff() {
        return this.adaptee.readStaffFile();
    }

    @Override
    public void removeStaffByName(List<original.StaffModel> list) {
        this.adaptee2.removeStaff(list);
    }

    @Override
    public void addNewStaff(List<original.StaffModel> list) {
        this.adaptee2.addingStaff(list);
    }

    @Override
    public void writeByRoom(List<original.StaffModel> list) {
        // sort by room
        for (int i = 0; i < list.size(); i++) {
            for (int j = 0; j < list.size() - 1 - i; j++) {
                if (list.get(j).getRoom().compareTo(list.get(j + 1).getRoom()) > 0) {
                    StaffModel temp = list.get(j);
                    list.set(j, list.get(j + 1));
                    list.set(j + 1, temp);
                }
            }
        }
        adaptee.writeStaffFile(list);
    }

    @Override
    public void writeByName(List<original.StaffModel> list) {
        // sort by name
        for (int i = 0; i < list.size(); i++) {
            for (int j = 0; j < list.size() - 1 - i; j++) {
                if (list.get(j).getName().compareTo(list.get(j + 1).getName()) > 0) {
                    StaffModel temp = list.get(j);
                    list.set(j, list.get(j + 1));
                    list.set(j + 1, temp);
                }
            }
        }
        adaptee.writeStaffFile(list);
    }

    @Override
    public void listAllStaff(List<original.StaffModel> list) {
        adaptee.printStaffFile(list);
    }
}
